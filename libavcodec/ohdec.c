/*
 * This file is part of FFmpeg.
 *
 * Copyright (c) 2025 Zhao Zhili <quinkblack@foxmail.com>
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config_components.h"

#include <errno.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <time.h>
#include <multimedia/player_framework/native_avcapability.h>
#include <multimedia/player_framework/native_avcodec_videodecoder.h>
#include <native_buffer/native_buffer.h>

#include "libavutil/fifo.h"
#include "libavutil/dovi_meta.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_oh.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/thread.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "dovi_rpu.h"
#include "get_bits.h"
#include "h2645_parse.h"
#include "h2645_sei.h"
#include "hevc/hevc.h"
#include "sei.h"
#include "hwconfig.h"
#include "ohcodec.h"
#include "ohcodec_buffer.h"

#define OH_HDR_QUEUE_SIZE 64

/*
 * HarmonyOS smart fluency (intelligent frame retention), available since
 * API 26.0.0. The OH_MD_KEY_VIDEO_DECODER_FRAME_RETENTION_MODE/SPEED keys
 * and the OH_FrameRetentionMode enum come from native_avcodec_base.h (pulled
 * in via native_avcodec_videodecoder.h). The two key symbols are referenced
 * weakly so that libmpv stays loadable on systems older than API 26.0.0; on
 * those devices the feature is simply inactive.
 */
extern const char *OH_MD_KEY_VIDEO_DECODER_FRAME_RETENTION_MODE
    __attribute__((weak));
extern const char *OH_MD_KEY_VIDEO_DECODER_SPEED
    __attribute__((weak));

/* Engage ADAPTIVE only when fps * speed exceeds this effective frame rate. */
#define OH_FLUENCY_EFFECTIVE_FPS_LIMIT 90.0

static bool oh_fluency_supported(void)
{
    /* NB: must compare the ADDRESS of the weak symbols. Reading the
     * variable itself dereferences a NULL GOT entry and crashes on
     * systems older than API 26.0.0. */
    return &OH_MD_KEY_VIDEO_DECODER_FRAME_RETENTION_MODE != NULL &&
           &OH_MD_KEY_VIDEO_DECODER_SPEED != NULL;
}

typedef struct OHCodecDecQueueItem {
    uint32_t index;
    OH_AVBuffer *buffer;
    uint64_t generation;
} OHCodecDecQueueItem;

// Per-picture HDR metadata recovered from the bitstream (see
// oh_decode_parse_hdr), held until the decoder outputs the picture.
typedef struct OHHdrFrame {
    int64_t pts;
    AVBufferRef *dovi_rpu;
    AVBufferRef *dovi_metadata;
    AVBufferRef *hdr_vivid;
    AVBufferRef *hdr_plus;
} OHHdrFrame;

typedef struct OHCodecDecRef {
    OH_AVCodec *dec;
    atomic_uint_fast64_t generation;
} OHCodecDecRef;

typedef struct OHCodecDecContext {
    AVClass *avclass;
    OH_AVCodec *dec;
    /* A reference count to dec. Each hardware frame has a reference count to
     * dec. dec will be destroyed only after oh_decode_close and all hardware
     * frames have been released.
     */
    AVBufferRef *dec_ref;

    AVMutex input_mutex;
    AVCond input_cond;
    AVFifo *input_queue;

    AVMutex output_mutex;
    AVCond output_cond;
    AVFifo *output_queue;

    AVPacket pkt;

    DOVIContext dovi_ctx;
    H2645Packet hdr_pkt;
    OHHdrFrame hdr_frames[OH_HDR_QUEUE_SIZE];
    int nb_hdr_frames;
    // Static HDR10 metadata from SEI, in effect for every output frame.
    H2645SEIMasteringDisplay mastering_display;
    H2645SEIContentLight content_light;

    int decode_status;
    bool eof_sent;
    atomic_bool accepting_callbacks;

    bool output_to_window;
    bool got_stream_info;
    int width;
    int height;
    int stride;
    int slice_height;
    OH_AVPixelFormat pix_fmt;
    int bit_depth;

    char *name;
    int allow_sw;

    /* Smart fluency state: last applied OH_FrameRetentionMode (-1 if none),
     * the speed last forwarded to the decoder in ADAPTIVE mode, and whether
     * the one-time diagnostic line has been logged. */
    int fluency_mode;
    double fluency_speed;
    bool fluency_logged;

    /* Variable refresh rate (VRR, API 15): last applied enable state and the
     * content frame rate last forwarded to the decoder. vrr_enabled == -1
     * means "not yet applied". */
    int vrr_enabled;
    int32_t vrr_fps;
} OHCodecDecContext;

struct AVOHCodecBuffer {
    uint32_t index;
    AVBufferRef *dec_ref;
    uint64_t generation;
    atomic_bool released;
};

static OHCodecDecRef *oh_decode_ref(const OHCodecDecContext *s)
{
    return (OHCodecDecRef *)s->dec_ref->data;
}

static uint64_t oh_decode_generation(const OHCodecDecContext *s)
{
    return atomic_load_explicit(&oh_decode_ref(s)->generation,
                                memory_order_acquire);
}

static void oh_decode_invalidate_outputs(OHCodecDecContext *s)
{
    atomic_fetch_add_explicit(&oh_decode_ref(s)->generation, 1,
                              memory_order_acq_rel);
}

static void oh_decode_unref_hdr(OHHdrFrame *hdr)
{
    av_buffer_unref(&hdr->dovi_rpu);
    av_buffer_unref(&hdr->dovi_metadata);
    av_buffer_unref(&hdr->hdr_vivid);
    av_buffer_unref(&hdr->hdr_plus);
}

static void oh_decode_clear_hdr(OHCodecDecContext *s)
{
    for (int i = 0; i < s->nb_hdr_frames; i++)
        oh_decode_unref_hdr(&s->hdr_frames[i]);
    s->nb_hdr_frames = 0;
}

/*
 * The OHOS decoder never surfaces SEI, so on the hardware path all in-band HDR
 * metadata would otherwise be lost: the HDR10 mastering display colour volume
 * and content light level (SEI 137/144), without which a player can only
 * assume the 10000 nit PQ ceiling, and the dynamic HDR Vivid / HDR10+ payloads
 * (ITU-T T.35). The NAL split performed for the Dolby Vision RPU already hands
 * us every NAL of the packet, so walk the prefix SEI ones here and let the
 * shared h2645 parser decode them.
 *
 * Mirrors the message loop of decode_nal_sei_message() in hevc/sei.c; the
 * parameter-set dependent SEI types are deliberately not handled, which is why
 * ff_h2645_sei_message_decode() is called directly instead of
 * ff_hevc_decode_nal_sei() (the latter needs an HEVCParamSets this decoder does
 * not maintain).
 */
static void oh_decode_parse_sei(AVCodecContext *avctx, const H2645NAL *nal,
                                OHHdrFrame *hdr)
{
    OHCodecDecContext *s = avctx->priv_data;
    H2645SEI sei = { 0 };
    GetByteContext gb;

    // nal->data is the RBSP payload; skip the two-byte HEVC NAL header.
    if (nal->size <= 2)
        return;
    bytestream2_init(&gb, nal->data + 2, nal->size - 2);

    while (bytestream2_get_bytes_left(&gb) > 0) {
        GetByteContext msg_gbyte;
        GetBitContext msg_gb;
        int payload_type = 0, payload_size = 0, byte = 0xFF;

        while (byte == 0xFF) {
            if (bytestream2_get_bytes_left(&gb) < 2 ||
                payload_type > INT_MAX - 255)
                goto done;
            byte          = bytestream2_get_byteu(&gb);
            payload_type += byte;
        }
        byte = 0xFF;
        while (byte == 0xFF) {
            if (bytestream2_get_bytes_left(&gb) < 1 + payload_size)
                goto done;
            byte          = bytestream2_get_byteu(&gb);
            payload_size += byte;
        }
        if (bytestream2_get_bytes_left(&gb) < payload_size)
            goto done;
        bytestream2_init(&msg_gbyte, gb.buffer, payload_size);
        if (init_get_bits8(&msg_gb, gb.buffer, payload_size) < 0)
            goto done;
        bytestream2_skipu(&gb, payload_size);

        // Only the HDR payloads; decoding anything else here would duplicate
        // work the OHOS decoder already did.
        switch (payload_type) {
        case SEI_TYPE_USER_DATA_REGISTERED_ITU_T_T35:
        case SEI_TYPE_MASTERING_DISPLAY_COLOUR_VOLUME:
        case SEI_TYPE_CONTENT_LIGHT_LEVEL_INFO:
            ff_h2645_sei_message_decode(&sei, payload_type, AV_CODEC_ID_HEVC,
                                        &msg_gb, &msg_gbyte, avctx);
            break;
        }
    }

done:
    // The static metadata describes the whole coded video sequence (encoders
    // repeat it at every IRAP), so it stays in effect until replaced.
    if (sei.mastering_display.present)
        s->mastering_display = sei.mastering_display;
    if (sei.content_light.present)
        s->content_light = sei.content_light;
    // The dynamic payloads belong to this picture alone. Steal them; a later
    // SEI NAL in the same packet wins.
    if (sei.dynamic_hdr_vivid.info) {
        av_buffer_unref(&hdr->hdr_vivid);
        hdr->hdr_vivid = sei.dynamic_hdr_vivid.info;
        sei.dynamic_hdr_vivid.info = NULL;
    }
    if (sei.dynamic_hdr_plus.info) {
        av_buffer_unref(&hdr->hdr_plus);
        hdr->hdr_plus = sei.dynamic_hdr_plus.info;
        sei.dynamic_hdr_plus.info = NULL;
    }
    ff_h2645_sei_reset(&sei);
}

// A malformed RPU is dropped without costing the packet's other metadata;
// only allocation failures are reported.
static int oh_decode_parse_rpu(AVCodecContext *avctx, const H2645NAL *nal,
                               OHHdrFrame *hdr)
{
    OHCodecDecContext *s = avctx->priv_data;
    AVDOVIMetadata *dovi = NULL;
    int ret;

    hdr->dovi_rpu = av_buffer_alloc(nal->raw_size - 2);
    if (!hdr->dovi_rpu)
        return AVERROR(ENOMEM);
    memcpy(hdr->dovi_rpu->data, nal->raw_data + 2, nal->raw_size - 2);

    ret = ff_dovi_rpu_parse(&s->dovi_ctx, nal->data + 2, nal->size - 2,
                            avctx->err_recognition);
    if (ret < 0) {
        av_buffer_unref(&hdr->dovi_rpu);
        return 0;
    }
    ret = ff_dovi_get_metadata(&s->dovi_ctx, &dovi);
    if (ret <= 0)
        return ret;
    hdr->dovi_metadata = av_buffer_create((uint8_t *)dovi, ret, NULL, NULL, 0);
    if (!hdr->dovi_metadata) {
        av_free(dovi);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int oh_decode_parse_hdr(AVCodecContext *avctx, const AVPacket *pkt)
{
    OHCodecDecContext *s = avctx->priv_data;
    OHHdrFrame hdr = { 0 };
    const H2645NAL *rpu_nal = NULL;
    size_t sd_size;
    const uint8_t *sd;
    int ret;

    if (avctx->codec_id != AV_CODEC_ID_HEVC || !pkt->size)
        return 0;
    sd = av_packet_get_side_data(pkt, AV_PKT_DATA_DOVI_CONF, &sd_size);
    if (sd && sd_size >= sizeof(s->dovi_ctx.cfg))
        s->dovi_ctx.cfg = *(const AVDOVIDecoderConfigurationRecord *)sd;
    ret = ff_h2645_packet_split(&s->hdr_pkt, pkt->data, pkt->size, avctx, 0,
                                AV_CODEC_ID_HEVC, H2645_FLAG_SMALL_PADDING);
    if (ret < 0)
        return 0;

    for (int i = 0; i < s->hdr_pkt.nb_nals; i++) {
        const H2645NAL *nal = &s->hdr_pkt.nals[i];
        if (nal->nuh_layer_id)
            continue;
        if (nal->type == HEVC_NAL_SEI_PREFIX)
            oh_decode_parse_sei(avctx, nal, &hdr);
        else if (nal->type == HEVC_NAL_UNSPEC62 && !nal->temporal_id &&
                 nal->size > 2 && nal->raw_size > 2)
            rpu_nal = nal; // the last RPU of the packet wins
    }

    if (rpu_nal && (ret = oh_decode_parse_rpu(avctx, rpu_nal, &hdr)) < 0)
        goto fail;

    if (!hdr.dovi_rpu && !hdr.dovi_metadata && !hdr.hdr_vivid && !hdr.hdr_plus)
        return 0;

    if (s->nb_hdr_frames == OH_HDR_QUEUE_SIZE) {
        oh_decode_unref_hdr(&s->hdr_frames[0]);
        memmove(&s->hdr_frames[0], &s->hdr_frames[1],
                sizeof(s->hdr_frames[0]) * (OH_HDR_QUEUE_SIZE - 1));
        s->nb_hdr_frames--;
    }
    hdr.pts = pkt->pts == AV_NOPTS_VALUE ? AV_NOPTS_VALUE :
              av_rescale_q(pkt->pts, avctx->pkt_timebase, AV_TIME_BASE_Q);
    s->hdr_frames[s->nb_hdr_frames++] = hdr;
    return 0;

fail:
    oh_decode_unref_hdr(&hdr);
    return ret;
}

// Moves ownership of *buf onto the frame. On success *buf is cleared; on
// failure it is released. Either way the caller must not attach it again.
static int oh_decode_attach_side_data(AVFrame *frame, AVBufferRef **buf,
                                      enum AVFrameSideDataType type)
{
    if (!*buf)
        return 0;
    if (!av_frame_new_side_data_from_buf(frame, type, *buf)) {
        av_buffer_unref(buf);
        return AVERROR(ENOMEM);
    }
    *buf = NULL;
    return 0;
}

// Attaches the HDR metadata that applies to an output frame: the static HDR10
// metadata in effect, plus the per-picture payloads oh_decode_parse_hdr()
// queued under its pts.
static int oh_decode_attach_hdr(AVCodecContext *avctx, AVFrame *frame,
                                int64_t pts)
{
    OHCodecDecContext *s = avctx->priv_data;
    // The shared SEI exporter validates the values, reorders the primaries and
    // honours side_data_prefer_packet against container (mdcv/clli) metadata
    // that ff_decode_frame_props() may already have put on the frame.
    H2645SEI sei = {
        .mastering_display = s->mastering_display,
        .content_light     = s->content_light,
    };
    int found = -1;
    int ret = ff_h2645_sei_to_frame(frame, &sei, avctx->codec_id, avctx,
                                    NULL, 0, 0, 0);
    if (ret < 0)
        return ret;

    for (int i = 0; i < s->nb_hdr_frames; i++)
        if (s->hdr_frames[i].pts == pts) { found = i; break; }
    if (found < 0)
        return 0;
    OHHdrFrame hdr = s->hdr_frames[found];
    memmove(&s->hdr_frames[found], &s->hdr_frames[found + 1],
            sizeof(s->hdr_frames[0]) * (s->nb_hdr_frames - found - 1));
    s->nb_hdr_frames--;

    ret = oh_decode_attach_side_data(frame, &hdr.dovi_rpu,
                                     AV_FRAME_DATA_DOVI_RPU_BUFFER);
    if (ret >= 0)
        ret = oh_decode_attach_side_data(frame, &hdr.dovi_metadata,
                                         AV_FRAME_DATA_DOVI_METADATA);
    if (ret >= 0)
        ret = oh_decode_attach_side_data(frame, &hdr.hdr_vivid,
                                         AV_FRAME_DATA_DYNAMIC_HDR_VIVID);
    if (ret >= 0)
        ret = oh_decode_attach_side_data(frame, &hdr.hdr_plus,
                                         AV_FRAME_DATA_DYNAMIC_HDR_PLUS);

    // Anything not attached (short-circuited by an error, or never present)
    // is owned by us and must be released here.
    oh_decode_unref_hdr(&hdr);
    return ret;
}

static void oh_decode_release(void *opaque, uint8_t *data)
{
    OHCodecDecRef *ref = (OHCodecDecRef *)data;
    OH_AVCodec *dec = ref->dec;
    OH_AVErrCode err = OH_VideoDecoder_Destroy(dec);
    if (err == AV_ERR_OK)
        av_log(NULL, AV_LOG_DEBUG, "Destroy decoder success\n");
    else
        av_log(NULL, AV_LOG_ERROR, "Destroy decoder failed, %d, %s\n",
               err, av_err2str(ff_oh_err_to_ff_err(err)));
}

static int oh_decode_create(OHCodecDecContext *s, AVCodecContext *avctx)
{
    const char *name = s->name;

    if (!name) {
        const char *mime = ff_oh_mime(avctx->codec_id, avctx);
        if (!mime)
            return AVERROR_BUG;
        OH_AVCapability *cap = OH_AVCodec_GetCapabilityByCategory(mime, false, HARDWARE);
        if (!cap) {
            if (!s->allow_sw) {
                av_log(avctx, AV_LOG_ERROR, "Failed to get hardware codec %s\n", mime);
                return AVERROR_EXTERNAL;
            }
            av_log(avctx, AV_LOG_WARNING,
                   "Failed to get hardware codec %s, try software backend\n", mime);
            cap = OH_AVCodec_GetCapabilityByCategory(mime, false, SOFTWARE);
            if (!cap) {
                av_log(avctx, AV_LOG_ERROR, "Failed to get software codec %s\n", mime);
                return AVERROR_EXTERNAL;
            }
        }
        name = OH_AVCapability_GetName(cap);
        if (!name)
            return AVERROR_EXTERNAL;
    }

    s->dec = OH_VideoDecoder_CreateByName(name);
    if (!s->dec) {
        av_log(avctx, AV_LOG_ERROR, "Create decoder with name %s failed\n", name);
        return AVERROR_EXTERNAL;
    }
    av_log(avctx, AV_LOG_DEBUG, "Create decoder %s success\n", name);

    OHCodecDecRef *ref = av_mallocz(sizeof(*ref));
    if (!ref)
        return AVERROR(ENOMEM);
    ref->dec = s->dec;
    atomic_init(&ref->generation, 0);
    s->dec_ref = av_buffer_create((uint8_t *)ref, sizeof(*ref),
                                  oh_decode_release, NULL, 0);
    if (!s->dec_ref) {
        av_free(ref);
        OH_VideoDecoder_Destroy(s->dec);
        s->dec = NULL;
        return AVERROR(ENOMEM);
    }

    return 0;
}

static int oh_decode_init_hwframes(OHCodecDecContext *s, AVCodecContext *avctx,
                                   enum AVPixelFormat sw_format)
{
    AVHWFramesContext *frames;
    int ret;

    if (avctx->hw_frames_ctx) {
        frames = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
        if (frames->format == AV_PIX_FMT_OHCODEC &&
            frames->sw_format == sw_format && frames->width == s->width &&
            frames->height == s->height)
            return 0;
        av_buffer_unref(&avctx->hw_frames_ctx);
    }
    avctx->hw_frames_ctx = av_hwframe_ctx_alloc(avctx->hw_device_ctx);
    if (!avctx->hw_frames_ctx)
        return AVERROR(ENOMEM);
    frames = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
    frames->format = AV_PIX_FMT_OHCODEC;
    frames->sw_format = sw_format;
    frames->width = s->width;
    frames->height = s->height;
    ret = av_hwframe_ctx_init(avctx->hw_frames_ctx);
    if (ret < 0)
        av_buffer_unref(&avctx->hw_frames_ctx);
    return ret;
}

static int oh_decode_set_format(OHCodecDecContext *s, AVCodecContext *avctx)
{
    int ret;
    OHNativeWindow *window = NULL;

    if (avctx->hw_device_ctx) {
        AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)(avctx->hw_device_ctx->data);
        if (device_ctx->type == AV_HWDEVICE_TYPE_OHCODEC) {
            AVOHCodecDeviceContext *dev = device_ctx->hwctx;
            window = dev->native_window;
            s->output_to_window = window != NULL;
            if (!s->output_to_window) {
                av_log(avctx, AV_LOG_ERROR,
                       "OHCodec hardware decoding requires a NativeWindow\n");
                return AVERROR(EINVAL);
            }
        } else {
            av_log(avctx, AV_LOG_WARNING, "Ignore invalid hw device type %s\n",
                   av_hwdevice_get_type_name(device_ctx->type));
        }
    }

    if (avctx->width <= 0 || avctx->height <= 0) {
        av_log(avctx, AV_LOG_ERROR,
               "Invalid width/height (%dx%d), width and height are mandatory for ohcodec\n",
               avctx->width, avctx->height);
        return AVERROR(EINVAL);
    }

    OH_AVFormat *format = OH_AVFormat_Create();
    if (!format)
        return AVERROR(ENOMEM);

    OH_AVFormat_SetIntValue(format, OH_MD_KEY_WIDTH, avctx->width);
    OH_AVFormat_SetIntValue(format, OH_MD_KEY_HEIGHT, avctx->height);
    if (!s->output_to_window)
        OH_AVFormat_SetIntValue(format, OH_MD_KEY_PIXEL_FORMAT,
                                AV_PIXEL_FORMAT_NV12);
    else
        OH_AVFormat_SetIntValue(format, OH_MD_KEY_PIXEL_FORMAT,
                                AV_PIXEL_FORMAT_SURFACE_FORMAT);
    if (oh_fluency_supported()) {
        /* Smart fluency baseline: start in full passthrough. The runtime
         * path switches to ADAPTIVE when fps * speed exceeds 120. */
        OH_AVFormat_SetIntValue(format, OH_MD_KEY_VIDEO_DECODER_FRAME_RETENTION_MODE,
                                OH_FRAME_RETENTION_MODE_FULL);
    }
    OH_AVErrCode err = OH_VideoDecoder_Configure(s->dec, format);
    OH_AVFormat_Destroy(format);
    if (err != AV_ERR_OK) {
        ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_ERROR, "Decoder configure failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }
    if (oh_fluency_supported())
        s->fluency_mode = OH_FRAME_RETENTION_MODE_FULL;

    if (s->output_to_window) {
        err = OH_VideoDecoder_SetSurface(s->dec, window);
        if (err != AV_ERR_OK) {
            ret = ff_oh_err_to_ff_err(err);
            av_log(avctx, AV_LOG_ERROR, "Set surface failed, %d, %s\n",
                   err, av_err2str(ret));
            return ret;
        }
    }

    return 0;
}

static void oh_decode_on_err(OH_AVCodec *codec, int32_t err, void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;

    // Careful on the lock order.
    // Always lock input first.
    ff_mutex_lock(&s->input_mutex);
    ff_mutex_lock(&s->output_mutex);
    s->decode_status = ff_oh_err_to_ff_err(err);
    ff_mutex_unlock(&s->output_mutex);
    ff_mutex_unlock(&s->input_mutex);

    ff_cond_signal(&s->output_cond);
    ff_cond_signal(&s->input_cond);
}

static void oh_decode_on_stream_changed(OH_AVCodec *codec, OH_AVFormat *format,
                                        void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;
    int32_t n, graphic_format;
    double d;

    if (!OH_AVFormat_GetIntValue(format, OH_MD_KEY_VIDEO_PIC_WIDTH, &s->width) ||
        !OH_AVFormat_GetIntValue(format, OH_MD_KEY_VIDEO_PIC_HEIGHT, &s->height)) {
        av_log(avctx, AV_LOG_ERROR, "Get dimension info from format failed\n");
        goto out;
    }

    if (ff_set_dimensions(avctx, s->width, s->height) < 0)
        goto out;

    OH_AVFormat_GetIntValue(format, OH_MD_KEY_VIDEO_STRIDE, &s->stride);
    OH_AVFormat_GetIntValue(format, OH_MD_KEY_VIDEO_SLICE_HEIGHT, &s->slice_height);
    if (!s->output_to_window && (s->stride <= 0 || s->slice_height <= 0)) {
        av_log(avctx, AV_LOG_ERROR,
               "Buffer stride (%d) or slice height (%d) is invalid\n",
               s->stride, s->slice_height);
        goto out;
    }

    s->bit_depth = avctx->bits_per_raw_sample > 8 ? avctx->bits_per_raw_sample : 8;
    if (avctx->profile == AV_PROFILE_HEVC_MAIN_10 ||
        avctx->profile == AV_PROFILE_H264_HIGH_10)
        s->bit_depth = 10;
    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_BITS_PER_CODED_SAMPLE, &n) && n > 0)
        s->bit_depth = n;
    // Surface output reports its actual allocation format separately from
    // OH_MD_KEY_PIXEL_FORMAT (which remains NV12 for P010 on current codecs).
    // This key is present in the decoder's output description on API 20.
    if (OH_AVFormat_GetIntValue(format, "video_graphic_pixel_format",
                                &graphic_format) &&
        (graphic_format == NATIVEBUFFER_PIXEL_FMT_YCBCR_P010 ||
         graphic_format == NATIVEBUFFER_PIXEL_FMT_YCRCB_P010))
        s->bit_depth = 10;

    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_PIXEL_FORMAT, &n)) {
        enum AVPixelFormat sw_format;

        s->pix_fmt = n;
        /* When use output_to_window, the returned format is the memory
         * layout of hardware frame, not AV_PIXEL_FORMAT_SURFACE_FORMAT as
         * expected.
         */
        if (s->output_to_window) {
            avctx->pix_fmt = AV_PIX_FMT_OHCODEC;
            sw_format = s->bit_depth > 8 ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
            avctx->sw_pix_fmt = sw_format;
            if (oh_decode_init_hwframes(s, avctx, sw_format) < 0) {
                av_log(avctx, AV_LOG_ERROR,
                       "Failed to initialize OHCodec hardware frames\n");
                goto out;
            }
        } else {
            avctx->pix_fmt = ff_oh_pix_to_ff_pix(s->pix_fmt);
        }
        // Check whether this pixel format is supported
        if (avctx->pix_fmt == AV_PIX_FMT_NONE) {
            av_log(avctx, AV_LOG_ERROR, "Unsupported OH_AVPixelFormat %d\n",
                   n);
            goto out;
        }
    } else if (s->output_to_window) {
        enum AVPixelFormat sw_format = s->bit_depth > 8
                                     ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
        avctx->pix_fmt = AV_PIX_FMT_OHCODEC;
        avctx->sw_pix_fmt = sw_format;
        if (oh_decode_init_hwframes(s, avctx, sw_format) < 0)
            goto out;
    } else {
        av_log(avctx, AV_LOG_ERROR, "Failed to get pixel format\n");
        goto out;
    }

    if (OH_AVFormat_GetIntValue(format,
                                OH_MD_KEY_MATRIX_COEFFICIENTS,
                                &n))
        avctx->colorspace = n;
    if (OH_AVFormat_GetIntValue(format,
                                OH_MD_KEY_COLOR_PRIMARIES,
                                &n))
        avctx->color_primaries = n;
    if (OH_AVFormat_GetIntValue(format,
                                OH_MD_KEY_TRANSFER_CHARACTERISTICS,
                                &n))
        avctx->color_trc = n;
    if (OH_AVFormat_GetIntValue(format,
                                OH_MD_KEY_RANGE_FLAG,
                                &n))
        avctx->color_range = n ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;

    if (OH_AVFormat_GetDoubleValue(format, OH_MD_KEY_VIDEO_SAR, &d)) {
        AVRational sar = av_d2q(d, 4096 * 4);
        ff_set_sar(avctx, sar);
    }

    s->got_stream_info = true;

    return;
out:
    av_log(avctx, AV_LOG_ERROR, "Invalid format from decoder: %s\n",
           OH_AVFormat_DumpInfo(format));
    oh_decode_on_err(codec, AV_ERR_UNKNOWN, userdata);
}

static void oh_decode_on_need_input(OH_AVCodec *codec, uint32_t index,
                                    OH_AVBuffer *buffer, void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;
    if (!atomic_load_explicit(&s->accepting_callbacks, memory_order_acquire))
        return;
    OHCodecDecQueueItem item = {
        index, buffer, oh_decode_generation(s),
    };

    ff_mutex_lock(&s->input_mutex);
    int ret = av_fifo_write(s->input_queue, &item, 1);
    if (ret >= 0)
        ff_cond_signal(&s->input_cond);
    ff_mutex_unlock(&s->input_mutex);

    if (ret < 0)
        oh_decode_on_err(codec, AV_ERR_NO_MEMORY, userdata);
}

static void oh_decode_on_output(OH_AVCodec *codec, uint32_t index,
                                OH_AVBuffer *buffer, void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;
    if (!atomic_load_explicit(&s->accepting_callbacks, memory_order_acquire))
        return;
    OHCodecDecQueueItem item = {
        index, buffer, oh_decode_generation(s),
    };

    ff_mutex_lock(&s->output_mutex);
    int ret = av_fifo_write(s->output_queue, &item, 1);
    if (ret >= 0)
        ff_cond_signal(&s->output_cond);
    ff_mutex_unlock(&s->output_mutex);

    if (ret < 0)
        oh_decode_on_err(codec, AV_ERR_NO_MEMORY, userdata);
}

static int oh_decode_start(OHCodecDecContext *s, AVCodecContext *avctx)
{
    int ret;
    OH_AVErrCode err;
    OH_AVCodecCallback cb = {
        .onError = oh_decode_on_err,
        .onStreamChanged = oh_decode_on_stream_changed,
        .onNeedInputBuffer = oh_decode_on_need_input,
        .onNewOutputBuffer = oh_decode_on_output,
    };

    err = OH_VideoDecoder_RegisterCallback(s->dec, cb, avctx);
    if (err != AV_ERR_OK) {
        ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_ERROR, "Register callback failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }
    err = OH_VideoDecoder_Prepare(s->dec);
    if (err != AV_ERR_OK) {
        ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_ERROR, "Prepare failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }
    atomic_store_explicit(&s->accepting_callbacks, true, memory_order_release);
    err = OH_VideoDecoder_Start(s->dec);
    if (err != AV_ERR_OK) {
        atomic_store_explicit(&s->accepting_callbacks, false,
                              memory_order_release);
        ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_ERROR, "Start failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }

    return 0;
}

static av_cold int oh_decode_init(AVCodecContext *avctx)
{
    OHCodecDecContext *s = avctx->priv_data;

    atomic_init(&s->accepting_callbacks, false);
    s->fluency_mode = -1;
    s->vrr_enabled = -1;
    s->dovi_ctx.logctx = avctx;
    if (avctx->codec_id == AV_CODEC_ID_HEVC) {
        const AVPacketSideData *sd =
            av_packet_side_data_get(avctx->coded_side_data,
                                    avctx->nb_coded_side_data,
                                    AV_PKT_DATA_DOVI_CONF);
        if (sd && sd->size >= sizeof(s->dovi_ctx.cfg))
            s->dovi_ctx.cfg =
                *(const AVDOVIDecoderConfigurationRecord *)sd->data;
    }

    // Initialize these fields first, so oh_decode_close can destroy them safely
    ff_mutex_init(&s->input_mutex, NULL);
    ff_cond_init(&s->input_cond, NULL);
    ff_mutex_init(&s->output_mutex, NULL);
    ff_cond_init(&s->output_cond, NULL);

    int ret = oh_decode_create(s, avctx);
    if (ret < 0)
        return ret;
    ret = oh_decode_set_format(s, avctx);
    if (ret < 0)
        return ret;

    size_t fifo_size = 16;
    s->input_queue = av_fifo_alloc2(fifo_size, sizeof(OHCodecDecQueueItem),
                                    AV_FIFO_FLAG_AUTO_GROW);
    s->output_queue = av_fifo_alloc2(fifo_size, sizeof(OHCodecDecQueueItem),
                                     AV_FIFO_FLAG_AUTO_GROW);
    if (!s->input_queue || !s->output_queue)
        return AVERROR(ENOMEM);

    ret = oh_decode_start(s, avctx);
    if (ret < 0)
        return ret;

    return 0;
}

static av_cold int oh_decode_close(AVCodecContext *avctx)
{
    OHCodecDecContext *s = avctx->priv_data;

    if (s->dec) {
        atomic_store_explicit(&s->accepting_callbacks, false,
                              memory_order_release);
        oh_decode_invalidate_outputs(s);
        /* Stop but don't destroy dec directly, to keep hardware frames on
         * the fly valid.
         */
        OH_AVErrCode err = OH_VideoDecoder_Stop(s->dec);
        if (err == AV_ERR_OK)
            av_log(avctx, AV_LOG_DEBUG, "Stop decoder success\n");
        else
            av_log(avctx, AV_LOG_ERROR, "Stop decoder failed, %d, %s\n",
                   err, av_err2str(ff_oh_err_to_ff_err(err)));
        s->dec = NULL;
        av_buffer_unref(&s->dec_ref);
    }

    av_packet_unref(&s->pkt);
    oh_decode_clear_hdr(s);
    ff_h2645_packet_uninit(&s->hdr_pkt);
    ff_dovi_ctx_unref(&s->dovi_ctx);

    ff_mutex_destroy(&s->input_mutex);
    ff_cond_destroy(&s->input_cond);
    av_fifo_freep2(&s->input_queue);

    ff_mutex_destroy(&s->output_mutex);
    ff_cond_destroy(&s->output_cond);
    av_fifo_freep2(&s->output_queue);

    return 0;
}

int av_ohcodec_release_buffer(AVOHCodecBuffer *buffer, int render)
{
    OHCodecDecRef *ref;
    OH_AVErrCode err;
    if (!buffer)
        return AVERROR(EINVAL);
    if (atomic_exchange_explicit(&buffer->released, true, memory_order_acq_rel))
        return render ? AVERROR(EALREADY) : 0;
    ref = (OHCodecDecRef *)buffer->dec_ref->data;
    if (buffer->generation != atomic_load_explicit(&ref->generation,
                                                    memory_order_acquire))
        return render ? AVERROR(ESTALE) : 0;
    err = render ? OH_VideoDecoder_RenderOutputBuffer(ref->dec, buffer->index)
                 : OH_VideoDecoder_FreeOutputBuffer(ref->dec, buffer->index);
    return ff_oh_err_to_ff_err(err);
}

int av_ohcodec_dec_set_fluency_state(AVCodecContext *avctx, double fps,
                                     double speed)
{
    OHCodecDecContext *s;
    OH_AVFormat *param;
    OH_AVErrCode err;
    int mode;

    if (!avctx || !avctx->codec ||
        ffcodec(avctx->codec)->init != oh_decode_init)
        return AVERROR(EINVAL);
    s = avctx->priv_data;

    mode = oh_fluency_supported() && fps > 0 && speed > 1 &&
           fps * speed > OH_FLUENCY_EFFECTIVE_FPS_LIMIT
         ? OH_FRAME_RETENTION_MODE_ADAPTIVE : OH_FRAME_RETENTION_MODE_FULL;
    if (!s->fluency_logged) {
        av_log(avctx, AV_LOG_INFO,
               "Smart fluency %s: fps %.6g, speed %.6g -> %s\n",
               oh_fluency_supported() ? "supported" : "unavailable (API < 26)",
               fps, speed,
               mode == OH_FRAME_RETENTION_MODE_ADAPTIVE ? "ADAPTIVE" : "FULL");
        s->fluency_logged = true;
    }
    if (!oh_fluency_supported() || !s->dec)
        return 0;
    /* The threshold cannot be evaluated without both parameters; keep the
     * current (baseline FULL) mode. */
    if (!(fps > 0) || !(speed > 0))
        return 0;

    if (mode == s->fluency_mode &&
        (mode != OH_FRAME_RETENTION_MODE_ADAPTIVE || speed == s->fluency_speed))
        return 0;

    param = OH_AVFormat_Create();
    if (!param)
        return AVERROR(ENOMEM);
    OH_AVFormat_SetIntValue(param, OH_MD_KEY_VIDEO_DECODER_FRAME_RETENTION_MODE,
                            mode);
    /* ADAPTIVE requires the current playback speed as calculation basis. */
    if (mode == OH_FRAME_RETENTION_MODE_ADAPTIVE)
        OH_AVFormat_SetDoubleValue(param, OH_MD_KEY_VIDEO_DECODER_SPEED, speed);
    err = OH_VideoDecoder_SetParameter(s->dec, param);
    OH_AVFormat_Destroy(param);
    if (err != AV_ERR_OK) {
        int ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_WARNING,
               "Set smart fluency mode %d failed, %d, %s\n",
               mode, err, av_err2str(ret));
        return ret;
    }

    av_log(avctx, AV_LOG_INFO, "Smart fluency %s applied: fps %.6g, speed %.6g\n",
           mode == OH_FRAME_RETENTION_MODE_ADAPTIVE ? "ADAPTIVE" : "FULL",
           fps, speed);
    s->fluency_mode = mode;
    s->fluency_speed = speed;
    return 0;
}

int av_ohcodec_dec_set_vrr(AVCodecContext *avctx, int32_t fps, int enable)
{
    OHCodecDecContext *s;
    OH_AVFormat *param;
    OH_AVErrCode err;
    int want;

    if (!avctx || !avctx->codec ||
        ffcodec(avctx->codec)->init != oh_decode_init)
        return AVERROR(EINVAL);
    s = avctx->priv_data;
    if (!s->dec)
        return 0;
    /* VRR only takes effect for hardware decode direct-to-surface; setting it
     * in buffer-copy mode is a harmless no-op, so skip the call entirely. */
    if (!s->output_to_window)
        return 0;
    /* Enabling requires a valid content frame rate (see OH_MD_KEY_FRAME_RATE
     * dependency in the VRR guide). Disabling needs none. */
    if (enable && fps <= 0)
        return 0;

    want = enable ? 1 : 0;
    if (want == s->vrr_enabled && (!enable || fps == s->vrr_fps))
        return 0;

    param = OH_AVFormat_Create();
    if (!param)
        return AVERROR(ENOMEM);
    if (enable) {
        OH_AVFormat_SetIntValue(param, OH_MD_KEY_FRAME_RATE, fps);
        OH_AVFormat_SetIntValue(param,
                                OH_MD_KEY_VIDEO_DECODER_OUTPUT_ENABLE_VRR, 1);
    } else {
        OH_AVFormat_SetIntValue(param,
                                OH_MD_KEY_VIDEO_DECODER_OUTPUT_ENABLE_VRR, 0);
    }
    err = OH_VideoDecoder_SetParameter(s->dec, param);
    OH_AVFormat_Destroy(param);
    if (err != AV_ERR_OK) {
        int ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_WARNING,
               "Set VRR enable=%d fps=%d failed, %d, %s\n",
               enable, fps, err, av_err2str(ret));
        return ret;
    }

    av_log(avctx, AV_LOG_INFO, "VRR %s: fps=%d\n",
           enable ? "enabled" : "disabled", fps);
    s->vrr_enabled = want;
    if (enable)
        s->vrr_fps = fps;
    return 0;
}

static void oh_buffer_release(void *opaque, uint8_t *data)
{
    AVOHCodecBuffer *buffer = opaque;
    if (!buffer)
        return;
    av_ohcodec_release_buffer(buffer, 0);

    av_buffer_unref(&buffer->dec_ref);
    av_free(buffer);
}

static int oh_decode_wrap_hw_buffer(AVCodecContext *avctx, AVFrame *frame,
                                    OHCodecDecQueueItem *output,
                                    const OH_AVCodecBufferAttr *attr)
{
    OHCodecDecContext *s = avctx->priv_data;

    frame->width = s->width;
    frame->height = s->height;
    int ret = ff_decode_frame_props(avctx, frame);
    if (ret < 0)
        return ret;

    frame->format = AV_PIX_FMT_OHCODEC;
    AVOHCodecBuffer *buffer = av_mallocz(sizeof(*buffer));
    if (!buffer)
        return AVERROR(ENOMEM);

    buffer->dec_ref = av_buffer_ref(s->dec_ref);
    if (!buffer->dec_ref) {
        oh_buffer_release(buffer, NULL);
        return AVERROR(ENOMEM);
    }

    buffer->index = output->index;
    buffer->generation = output->generation;
    atomic_init(&buffer->released, false);
    frame->buf[0] = av_buffer_create((uint8_t *)buffer, 1,
                                     oh_buffer_release,
                                     buffer, AV_BUFFER_FLAG_READONLY);
    if (!frame->buf[0]) {
        oh_buffer_release(buffer, NULL);
        return AVERROR(ENOMEM);
    }
    frame->hw_frames_ctx = av_buffer_ref(avctx->hw_frames_ctx);
    if (!frame->hw_frames_ctx) {
        av_buffer_unref(&frame->buf[0]);
        return AVERROR(ENOMEM);
    }
    // Opaque AVOHCodecBuffer token, never an addressable image plane.
    frame->data[3] = frame->buf[0]->data;
    frame->pts = av_rescale_q(attr->pts, AV_TIME_BASE_Q, avctx->pkt_timebase);
    frame->pkt_dts = AV_NOPTS_VALUE;

    return oh_decode_attach_hdr(avctx, frame, attr->pts);
}

static int oh_decode_wrap_sw_buffer(AVCodecContext *avctx, AVFrame *frame,
                                    OHCodecDecQueueItem *output,
                                    const OH_AVCodecBufferAttr *attr)
{
    OHCodecDecContext *s = avctx->priv_data;

    frame->format = avctx->pix_fmt;
    frame->width = s->width;
    frame->height = s->height;
    int ret = ff_get_buffer(avctx, frame, 0);
    if (ret < 0)
        return ret;

    frame->pts = av_rescale_q(attr->pts, AV_TIME_BASE_Q, avctx->pkt_timebase);
    frame->pkt_dts = AV_NOPTS_VALUE;

    uint8_t *p = OH_AVBuffer_GetAddr(output->buffer);
    if (!p) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get output buffer addr\n");
        return AVERROR_EXTERNAL;
    }

    uint8_t *src[4] = {0};
    int src_linesizes[4] = {0};

    ret = av_image_fill_linesizes(src_linesizes, frame->format, s->stride);
    if (ret < 0)
        return ret;
    ret = av_image_fill_pointers(src, frame->format, s->slice_height, p,
                                 src_linesizes);
    if (ret < 0)
        return ret;
    av_image_copy2(frame->data, frame->linesize, src, src_linesizes,
                   frame->format, frame->width, frame->height);

    OH_AVErrCode err = OH_VideoDecoder_FreeOutputBuffer(s->dec, output->index);
    if (err != AV_ERR_OK) {
        ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_ERROR, "FreeOutputBuffer failed, %d, %s\n", err,
               av_err2str(ret));
        return ret;
    }

    return oh_decode_attach_hdr(avctx, frame, attr->pts);
}

static int oh_decode_output_frame(AVCodecContext *avctx, AVFrame *frame,
                                  OHCodecDecQueueItem *output)
{
    OHCodecDecContext *s = avctx->priv_data;
    OH_AVCodecBufferAttr attr;

    OH_AVErrCode err = OH_AVBuffer_GetBufferAttr(output->buffer, &attr);
    if (err != AV_ERR_OK)
        return ff_oh_err_to_ff_err(err);

    if (attr.flags & AVCODEC_BUFFER_FLAGS_EOS) {
        av_log(avctx, AV_LOG_DEBUG, "Buffer flag eos\n");
        OH_VideoDecoder_FreeOutputBuffer(s->dec, output->index);
        return AVERROR_EOF;
    }

    if (!s->got_stream_info) {
        // This shouldn't happen, add a warning message.
        av_log(avctx, AV_LOG_WARNING,
               "decoder didn't notify stream info, try get format explicitly\n");

        OH_AVFormat *format = OH_VideoDecoder_GetOutputDescription(s->dec);
        if (!format) {
            av_log(avctx, AV_LOG_ERROR, "GetOutputDescription failed\n");
            return AVERROR_EXTERNAL;
        }

        oh_decode_on_stream_changed(s->dec, format, avctx);
        OH_AVFormat_Destroy(format);
        if (!s->got_stream_info)
            return AVERROR_EXTERNAL;
    }

    if (s->output_to_window)
        return oh_decode_wrap_hw_buffer(avctx, frame, output, &attr);
    return oh_decode_wrap_sw_buffer(avctx, frame, output, &attr);
}

static int oh_decode_send_pkt(AVCodecContext *avctx,
                              OHCodecDecQueueItem *input)
{
    OHCodecDecContext *s = avctx->priv_data;
    OH_AVErrCode err;
    int ret;

    if (!s->pkt.size && !s->eof_sent) {
        OH_AVCodecBufferAttr attr = {
            .flags = AVCODEC_BUFFER_FLAGS_EOS,
        };
        err = OH_AVBuffer_SetBufferAttr(input->buffer, &attr);
        if (err != AV_ERR_OK)
            return ff_oh_err_to_ff_err(err);
        err = OH_VideoDecoder_PushInputBuffer(s->dec, input->index);
        if (err != AV_ERR_OK)
            return ff_oh_err_to_ff_err(err);
        s->eof_sent = true;
        return 0;
    }

    uint8_t *p = OH_AVBuffer_GetAddr(input->buffer);
    int32_t n = OH_AVBuffer_GetCapacity(input->buffer);
    if (!p || n <= 0) {
        av_log(avctx, AV_LOG_ERROR,
               "Failed to get buffer addr (%p) or capacity (%d)\n",
               p, n);
        return AVERROR_EXTERNAL;
    }
    n = FFMIN(s->pkt.size, n);
    memcpy(p, s->pkt.data, n);

    OH_AVCodecBufferAttr attr = {
            .size = n,
            .offset = 0,
            .pts = av_rescale_q(s->pkt.pts, avctx->pkt_timebase,
                                AV_TIME_BASE_Q),
            .flags = (s->pkt.flags & AV_PKT_FLAG_KEY)
                     ? AVCODEC_BUFFER_FLAGS_SYNC_FRAME : 0,
    };

    err = OH_AVBuffer_SetBufferAttr(input->buffer, &attr);
    if (err != AV_ERR_OK) {
        ret = ff_oh_err_to_ff_err(err);
        return ret;
    }
    err = OH_VideoDecoder_PushInputBuffer(s->dec, input->index);
    if (err != AV_ERR_OK) {
        ret = ff_oh_err_to_ff_err(err);
        av_log(avctx, AV_LOG_ERROR, "Push input buffer failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }

    if (n < s->pkt.size) {
        s->pkt.size -= n;
        s->pkt.data += n;
    } else {
        av_packet_unref(&s->pkt);
    }

    return 0;
}

static int oh_decode_cond_wait(AVCond *cond, AVMutex *mutex)
{
    struct timespec deadline;
    if (clock_gettime(CLOCK_REALTIME, &deadline) < 0)
        return errno;
    deadline.tv_nsec += 100 * 1000 * 1000;
    if (deadline.tv_nsec >= 1000 * 1000 * 1000) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000 * 1000 * 1000;
    }
    return ff_cond_timedwait(cond, mutex, &deadline);
}

static int oh_decode_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    OHCodecDecContext *s = avctx->priv_data;

    while (1) {
        OHCodecDecQueueItem buffer = {0};
        int ret;

        // Try get output
        ff_mutex_lock(&s->output_mutex);
        while (!s->decode_status) {
            if (av_fifo_read(s->output_queue, &buffer, 1) >= 0)
                break;
            // Only wait after send EOF
            if (s->eof_sent && !s->decode_status) {
                if (oh_decode_cond_wait(&s->output_cond, &s->output_mutex) ==
                    ETIMEDOUT)
                    break;
            }
            else
                break;
        }

        ret = s->decode_status;
        ff_mutex_unlock(&s->output_mutex);

        // Got a frame
        if (buffer.buffer) {
            if (buffer.generation != oh_decode_generation(s))
                continue;
            return oh_decode_output_frame(avctx, frame, &buffer);
        }
        if (ret < 0)
            return ret;

        if (!s->pkt.size) {
            /* fetch new packet or eof */
            ret = ff_decode_get_packet(avctx, &s->pkt);
            if (ret < 0 && ret != AVERROR_EOF)
                return ret;
            if (ret >= 0 && (ret = oh_decode_parse_hdr(avctx, &s->pkt)) < 0)
                return ret;
        }

        // Wait input buffer
        ff_mutex_lock(&s->input_mutex);
        while (!s->decode_status) {
            if (av_fifo_read(s->input_queue, &buffer, 1) >= 0)
                break;
            if (oh_decode_cond_wait(&s->input_cond, &s->input_mutex) ==
                ETIMEDOUT)
                break;
        }

        ret = s->decode_status;
        ff_mutex_unlock(&s->input_mutex);

        if (ret < 0)
            return ret;

        if (!buffer.buffer)
            return AVERROR(EAGAIN);

        if (buffer.generation != oh_decode_generation(s))
            continue;

        ret = oh_decode_send_pkt(avctx, &buffer);
        if (ret < 0)
            return ret;
    }

    return AVERROR(EAGAIN);
}

static void oh_decode_flush(AVCodecContext *avctx)
{
    OHCodecDecContext *s = avctx->priv_data;

    atomic_store_explicit(&s->accepting_callbacks, false,
                          memory_order_release);
    oh_decode_invalidate_outputs(s);
    OH_VideoDecoder_Flush(s->dec);

    ff_mutex_lock(&s->input_mutex);
    ff_mutex_lock(&s->output_mutex);
    av_fifo_reset2(s->input_queue);
    av_fifo_reset2(s->output_queue);
    s->decode_status = 0;
    s->eof_sent = false;
    av_packet_unref(&s->pkt);
    oh_decode_clear_hdr(s);
    ff_dovi_ctx_flush(&s->dovi_ctx);
    ff_mutex_unlock(&s->output_mutex);
    ff_mutex_unlock(&s->input_mutex);

    atomic_store_explicit(&s->accepting_callbacks, true, memory_order_release);
    if (OH_VideoDecoder_Start(s->dec) != AV_ERR_OK)
        atomic_store_explicit(&s->accepting_callbacks, false,
                              memory_order_release);
    /* Retention mode persistence across Flush is undocumented. Invalidate
     * the cached state so the runtime path re-asserts it on the next frame. */
    s->fluency_mode = -1;
    /* Same reasoning for VRR: re-assert on the next frame. */
    s->vrr_enabled = -1;
}

static const AVCodecHWConfigInternal *const oh_hw_configs[] = {
    &(const AVCodecHWConfigInternal) {
        .public = {
            .pix_fmt = AV_PIX_FMT_OHCODEC,
            .methods = AV_CODEC_HW_CONFIG_METHOD_AD_HOC |
                       AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX,
            .device_type = AV_HWDEVICE_TYPE_OHCODEC,
        },
        .hwaccel = NULL,
    },
    NULL
};

#define OFFSET(x) offsetof(OHCodecDecContext, x)
#define VD (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM)
static const AVOption ohcodec_vdec_options[] = {
    {"codec_name", "Select codec by name",
         OFFSET(name), AV_OPT_TYPE_STRING, .flags = VD},
    {"allow_sw", "Allow software decoding",
         OFFSET(allow_sw), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, VD},
    {NULL}
};

#define DECLARE_OHCODEC_VCLASS(short_name)                                     \
  static const AVClass short_name##_oh_dec_class = {                           \
      .class_name = #short_name "_ohcodec",                                    \
      .item_name = av_default_item_name,                                       \
      .option = ohcodec_vdec_options,                                          \
      .version = LIBAVUTIL_VERSION_INT,                                        \
  };

#define DECLARE_OHCODEC_VDEC(short_name, full_name, codec_id, bsf)             \
  DECLARE_OHCODEC_VCLASS(short_name)                                           \
  const FFCodec ff_##short_name##_oh_decoder = {                               \
      .p.name = #short_name "_ohcodec",                                        \
      CODEC_LONG_NAME(full_name " OpenHarmony Codec"),                         \
      .p.type = AVMEDIA_TYPE_VIDEO,                                            \
      .p.id = codec_id,                                                        \
      .p.priv_class = &short_name##_oh_dec_class,                              \
      .priv_data_size = sizeof(OHCodecDecContext),                             \
      .init = oh_decode_init,                                                  \
      FF_CODEC_RECEIVE_FRAME_CB(oh_decode_receive_frame),                      \
      .flush = oh_decode_flush,                                                \
      .close = oh_decode_close,                                                \
      .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING |      \
                        AV_CODEC_CAP_HARDWARE,                                 \
      .caps_internal = FF_CODEC_CAP_INIT_CLEANUP,                              \
      .bsfs = bsf,                                                             \
      .hw_configs = oh_hw_configs,                                             \
      .p.wrapper_name = "ohcodec",                                             \
  };

#if CONFIG_H264_OH_DECODER
DECLARE_OHCODEC_VDEC(h264, "H.264", AV_CODEC_ID_H264, "h264_mp4toannexb")
#endif

#if CONFIG_HEVC_OH_DECODER
DECLARE_OHCODEC_VDEC(hevc, "H.265", AV_CODEC_ID_HEVC, "hevc_mp4toannexb")
#endif
