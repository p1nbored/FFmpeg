/*
 * OpenHarmony Codec public output-buffer API
 *
 * This file is part of FFmpeg.
 */

#ifndef AVCODEC_OHCODEC_BUFFER_H
#define AVCODEC_OHCODEC_BUFFER_H

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

#endif /* AVCODEC_OHCODEC_BUFFER_H */
