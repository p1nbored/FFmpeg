/*
 * OpenHarmony Codec public output-buffer API
 *
 * This file is part of FFmpeg.
 */

#ifndef AVCODEC_OHCODEC_BUFFER_H
#define AVCODEC_OHCODEC_BUFFER_H

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

#endif /* AVCODEC_OHCODEC_BUFFER_H */
