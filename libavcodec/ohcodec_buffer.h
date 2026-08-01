/*
 * OpenHarmony Codec public output-buffer API
 *
 * This file is part of FFmpeg.
 */

#ifndef AVCODEC_OHCODEC_BUFFER_H
#define AVCODEC_OHCODEC_BUFFER_H

#include <stdint.h>

struct AVCodecContext;

/** Opaque output token carried by AV_PIX_FMT_OHCODEC frames. */
typedef struct AVOHCodecBuffer AVOHCodecBuffer;

/**
 * Return an output buffer to the decoder exactly once.
 *
 * If render is non-zero, the decoder queues the buffer to its configured
 * Surface. Otherwise the buffer is discarded. Calls after the first one are
 * harmless and return success. A token invalidated by flush/stop is discarded
 * without calling the codec with a stale output index. A render request for
 * an already released token returns AVERROR(EALREADY); a render request for a
 * token invalidated by flush/stop returns AVERROR(ESTALE). Discard requests
 * remain idempotent.
 */
int av_ohcodec_release_buffer(AVOHCodecBuffer *buffer, int render);

/**
 * Update HarmonyOS smart fluency (intelligent frame retention) state of an
 * OHCodec video decoder. The capability requires API 26.0.0 or above; on
 * older systems this is a harmless no-op.
 *
 * FULL (transparent passthrough) mode is applied while fps * speed <= 120.
 * Above that threshold the decoder switches to ADAPTIVE mode, dropping
 * low-impact frames before decoding, and the current speed is forwarded to
 * the decoder as required by that mode.
 *
 * Safe to call repeatedly from the decoding thread; calls that don't change
 * the state are no-ops. fps <= 0 or speed <= 0 leaves the current mode
 * untouched.
 *
 * @param avctx an OHCodec decoder context (h264_ohcodec/hevc_ohcodec)
 * @param fps   video frame rate, <= 0 if unknown
 * @param speed current playback speed, > 0
 * @return 0 on success, AVERROR(EINVAL) if avctx is not an OHCodec decoder,
 *         or a negative error code if applying the parameter failed
 */
int av_ohcodec_dec_set_fluency_state(struct AVCodecContext *avctx, double fps,
                                     double speed);

/**
 * Enable/disable HarmonyOS video variable refresh rate (VRR) on an OHCodec
 * video decoder. Available since API 15.0.0; on unsupported platforms the
 * decoder ignores the keys without error.
 *
 * When enabled, the decoder signals the display pipeline to adjust the screen
 * refresh rate to match the content frame rate, dropping display-side frames
 * when the refresh rate is below the content rate to save power. Only takes
 * effect for hardware decode direct-to-surface (the OHCodec Surface path);
 * in buffer-copy mode the call is a no-op.
 *
 * Enabling requires a valid content frame rate (fps > 0), which is forwarded
 * via OH_MD_KEY_FRAME_RATE alongside the VRR switch. Safe to call repeatedly
 * from the decoding thread; unchanged state is a no-op.
 *
 * @param avctx  an OHCodec decoder context (h264_ohcodec/hevc_ohcodec)
 * @param fps    content frame rate, > 0 to enable (ignored when disabling)
 * @param enable non-zero to enable VRR, 0 to disable
 * @return 0 on success, AVERROR(EINVAL) if avctx is not an OHCodec decoder,
 *         or a negative error code if applying the parameter failed
 */
int av_ohcodec_dec_set_vrr(struct AVCodecContext *avctx, int32_t fps,
                           int enable);

#endif /* AVCODEC_OHCODEC_BUFFER_H */
