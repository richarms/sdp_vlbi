#!/usr/bin/env python3
import argparse, math, socket, struct, time
import numpy as np
from datetime import datetime, timezone

VDIF_HEADER_BYTES = 32
VDIF_VERSION = 1
BITS_PER_SAMPLE = 2
CHANNELS = 1
STATION_ID = "AA"
THREAD_ID = 0

def vdif_ref_epoch_info(unix_t):
    epoch0 = datetime(2000,1,1,tzinfo=timezone.utc).timestamp()
    secs = unix_t - epoch0
    half_year = 365 * 24 * 3600 / 2
    ref_epoch = int(secs // half_year) & 0x3F
    secs_from_ref = int(secs - ref_epoch * half_year)
    if secs_from_ref < 0: secs_from_ref = 0
    return ref_epoch, secs_from_ref

def build_vdif_header(secs_from_ref, ref_epoch, frame_within_sec, frame_len_bytes):
    log2ch = int(math.log2(CHANNELS))
    frame_len_8 = frame_len_bytes // 8
    w0 = secs_from_ref & 0x3FFFFFFF
    w1 = ((frame_within_sec & 0xFFFFFF) << 8) | ((log2ch & 0x1F) << 3) | ((BITS_PER_SAMPLE - 1) & 0x7)
    w2 = (ord(STATION_ID[0]) << 24) | (ord(STATION_ID[1]) << 16) | ((VDIF_VERSION & 0x1F) << 8) | (ref_epoch & 0x3F)
    w3 = (frame_len_8 & 0xFFFFFF) << 8
    w4 = (THREAD_ID & 0x3FF)
    h = bytearray(VDIF_HEADER_BYTES)
    struct.pack_into(">I", h,  0, w0)
    struct.pack_into(">I", h,  4, w1)
    struct.pack_into(">I", h,  8, w2)
    struct.pack_into(">I", h, 12, w3)
    struct.pack_into(">I", h, 16, w4)
    return h

def quantize_2bit_unsigned(x):
    thr = np.percentile(x, [25, 50, 75])
    return np.digitize(x, thr, right=True).astype(np.uint8)

def pack_2bit(q):
    pad = (-len(q)) % 4
    if pad: q = np.pad(q, (0, pad), mode="constant")
    q4 = q.reshape(-1, 4)
    b = ((q4[:,0] << 6) | (q4[:,1] << 4) | (q4[:,2] << 2) | q4[:,3]).astype(np.uint8)
    return b.tobytes()

def main():
    ap = argparse.ArgumentParser(description="VDIF UDP sender (std MTU, second-synced)")
    ap.add_argument("--ip", default="10.8.80.30")
    ap.add_argument("--port", type=int, default=50000)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--tone-hz", type=float, default=1_000_000.0)
    ap.add_argument("--noise-std", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=11176, help="integer frames/s")
    ap.add_argument("--seq", type=bool, default=True, help="prepend 8B seqno for jive5ab udps mode")
    ap.add_argument("--sndbuf", type=int, default=16*1024*1024)
    args = ap.parse_args()

    # Std MTU, udps seqno consumes 8 bytes: payload_bytes = 1472 - 8 - 32 = 1432
    vdif_payload_bytes = 1432 if args.seq else 1440
    samples_per_frame = vdif_payload_bytes * 4
    frame_bytes = VDIF_HEADER_BYTES + vdif_payload_bytes
    fps = int(args.fps)
    frame_duration = 1.0 / fps
    eff_sample_rate = samples_per_frame * fps

    print(f"Dest {args.ip}:{args.port}")
    print(f"udps-seq={'on' if args.seq else 'off'} | payload={vdif_payload_bytes}B | frame={frame_bytes}B | fps={fps} | Fs={eff_sample_rate/1e6:.6f} MS/s")

    # Wait for next integer second (first frame at boundary)
    while True:
        now = time.time()
        if now - math.floor(now) < 1e-3:
            break
        time.sleep(0.0002)

    # Ref epoch & seconds-from-ref at boundary
    boundary = math.floor(time.time())
    ref_epoch, secs_from_ref = vdif_ref_epoch_info(boundary)

    # One-frame time base
    t = np.arange(samples_per_frame, dtype=np.float64) / eff_sample_rate
    phase = 0.0
    phase_inc = 2*math.pi*args.tone_hz*frame_duration

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.sndbuf)

    n_frames = int(args.duration * fps)
    next_deadline = time.perf_counter()
    frame_within_sec = 0
    seqno = 0

    for _ in range(n_frames):
        signal = np.sin(2*math.pi*args.tone_hz*t + phase) + np.random.normal(0, args.noise_std, samples_per_frame)
        phase += phase_inc
        if phase >= 2*math.pi: phase -= 2*math.pi

        q = quantize_2bit_unsigned(signal)
        payload = pack_2bit(q)
        header = build_vdif_header(secs_from_ref, ref_epoch, frame_within_sec, frame_bytes)

        if args.seq:
            pkt = struct.pack(">Q", seqno) + header + payload  # 8B big-endian seqno
            seqno += 1
        else:
            pkt = header + payload

        sock.sendto(pkt, (args.ip, args.port))

        frame_within_sec += 1
        if frame_within_sec == fps:
            frame_within_sec = 0
            secs_from_ref += 1

        next_deadline += frame_duration
        sleep_time = next_deadline - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_deadline = time.perf_counter()

if __name__ == "__main__":
    main()

