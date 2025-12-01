'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { Track } from 'livekit-client';
import { AnimatePresence, motion } from 'motion/react';
import {
  type TrackReference,
  VideoTrack,
  useLocalParticipant,
  useTracks,
  useVoiceAssistant,
} from '@livekit/components-react';
import { cn } from '@/lib/utils';

const MotionContainer = motion.create('div');

// Cast to any to avoid strict typing mismatches with motion/react
const ANIMATION_TRANSITION: any = {
  type: 'spring',
  stiffness: 800,
  damping: 50,
  mass: 1,
};

const classNames = {
  grid: [
    'h-full w-full',
    'grid gap-x-4 place-content-center',
    'grid-cols-[1fr_1fr] grid-rows-[60px_1fr_60px]',
  ],
  agentChatOpenWithSecondTile: ['col-start-1 row-start-1', 'self-center justify-self-end'],
  agentChatOpenWithoutSecondTile: ['col-start-1 row-start-1', 'col-span-2', 'place-content-center'],
  agentChatClosed: ['col-start-1 row-start-1', 'col-span-2 row-span-3', 'place-content-center'],
  secondTileChatOpen: ['col-start-2 row-start-1', 'self-center justify-self-start'],
  secondTileChatClosed: ['col-start-2 row-start-3', 'place-content-end'],
};

export function useLocalTrackRef(source: Track.Source) {
  const { localParticipant } = useLocalParticipant();
  const publication = localParticipant.getTrackPublication(source);
  const trackRef = useMemo<TrackReference | undefined>(
    () => (publication ? { source, participant: localParticipant, publication } : undefined),
    [source, publication, localParticipant]
  );
  return trackRef;
}

/**
 * Custom ECG/Oscilloscope Visualizer (NEON)
 * Renders the audio waveform as a continuous line graph with neon glow.
 */
const ECGVisualizer = ({
  trackRef,
  className,
}: {
  trackRef?: TrackReference;
  className?: string;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trackRef?.publication?.track) return;

    const track = trackRef.publication.track;
    if (!track.mediaStreamTrack) return;

    const stream = new MediaStream([track.mediaStreamTrack]);
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioContext = new AudioContextClass();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);

    analyser.fftSize = 2048;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let animationId: number;

    const dpr = window.devicePixelRatio || 1;
    const resizeCanvas = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.scale(dpr, dpr);
    };

    // ensure canvas matches its CSS size for crisp neon
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      ctx.clearRect(0, 0, width, height);

      // subtle grid
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.03)';
      for (let gx = 0; gx < width; gx += 12) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
        ctx.stroke();
      }

      // neon waveform
      ctx.lineWidth = 2.2;
      const grad = ctx.createLinearGradient(0, 0, width, 0);
      grad.addColorStop(0, '#FF2D95'); // neon pink
      grad.addColorStop(0.5, '#FFFFFF');
      grad.addColorStop(1, '#18F3FF'); // neon cyan
      ctx.strokeStyle = grad;

      // glow
      ctx.shadowBlur = 14;
      ctx.shadowColor = '#FF2D95';

      ctx.beginPath();

      const sliceWidth = (width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0; // 0..2
        const y = (v * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();

      // soft outer glow line for extra retro effect
      ctx.lineWidth = 1;
      ctx.shadowBlur = 28;
      ctx.shadowColor = '#18F3FF';
      ctx.strokeStyle = 'rgba(24,243,255,0.15)';
      ctx.stroke();
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
      source.disconnect();
      analyser.disconnect();
      audioContext.close();
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [trackRef]);

  return <canvas ref={canvasRef} className={className} style={{ width: '100%', height: '100%' }} />;
};

interface TileLayoutProps {
  chatOpen: boolean;
}

export function TileLayout({ chatOpen }: TileLayoutProps) {
  const {
    state: agentState,
    audioTrack: agentAudioTrack,
    videoTrack: agentVideoTrack,
  } = useVoiceAssistant();
  const [screenShareTrack] = useTracks([Track.Source.ScreenShare]);
  const cameraTrack: TrackReference | undefined = useLocalTrackRef(Track.Source.Camera);

  const isCameraEnabled = cameraTrack && !cameraTrack.publication.isMuted;
  const isScreenShareEnabled = screenShareTrack && !screenShareTrack.publication.isMuted;
  const hasSecondTile = isCameraEnabled || isScreenShareEnabled;

  const animationDelay = chatOpen ? 0 : 0.15;
  const isAvatar = agentVideoTrack !== undefined;
  const videoWidth = agentVideoTrack?.publication.dimensions?.width ?? 0;
  const videoHeight = agentVideoTrack?.publication.dimensions?.height ?? 0;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-8 bottom-32 z-50 md:top-12 md:bottom-40">
      <div className="relative mx-auto h-full max-w-4xl px-4 md:px-0">
        <div className={cn(classNames.grid)}>
          {/* Agent Tile */}
          <div
            className={cn([
              'grid transition-all duration-500 ease-spring',
              !chatOpen && classNames.agentChatClosed,
              chatOpen && hasSecondTile && classNames.agentChatOpenWithSecondTile,
              chatOpen && !hasSecondTile && classNames.agentChatOpenWithoutSecondTile,
            ])}
          >
            <AnimatePresence mode="popLayout">
              {!isAvatar && (
                <MotionContainer
                  key="agent"
                  layoutId="agent"
                  initial={{ opacity: 0, scale: 0.8, filter: 'blur(6px)' }}
                  animate={{ opacity: 1, scale: chatOpen ? 1 : 1.15, filter: 'blur(0px)' }}
                  transition={( { ...ANIMATION_TRANSITION, delay: animationDelay } as any )}
                  className={cn(
                    'relative overflow-hidden',
                    'bg-black/90 backdrop-blur-md',
                    'border border-transparent',
                    chatOpen ? 'h-[60px] w-[60px] rounded-lg' : 'h-[120px] w-[120px] rounded-xl'
                  )}
                  style={{
                    boxShadow: chatOpen
                      ? '0 0 18px rgba(255,45,149,0.18), inset 0 0 6px rgba(24,243,255,0.03)'
                      : '0 0 30px rgba(24,243,255,0.08), 0 0 24px rgba(255,45,149,0.06)',
                    border: '1px solid rgba(255,45,149,0.12)',
                  }}
                >
                  <div
                    className="absolute inset-0 z-0 opacity-08"
                    style={{
                      background:
                        'radial-gradient(circle at 10% 10%, rgba(255,45,149,0.06), transparent 8%), radial-gradient(circle at 90% 80%, rgba(24,243,255,0.04), transparent 12%)',
                    }}
                  />
                  <ECGVisualizer
                    trackRef={agentAudioTrack}
                    className="relative z-10 h-full w-full"
                  />
                </MotionContainer>
              )}

              {isAvatar && (
                <MotionContainer
                  key="avatar"
                  layoutId="avatar"
                  initial={{ scale: 1, opacity: 1 }}
                  animate={{
                    borderRadius: chatOpen ? 8 : 12,
                  }}
                  transition={( { ...ANIMATION_TRANSITION, delay: animationDelay } as any )}
                  className={cn(
                    'relative overflow-hidden bg-black',
                    chatOpen ? 'h-[60px] w-[60px]' : 'h-auto w-full max-w-[400px] aspect-video'
                  )}
                  style={{
                    border: '1px solid rgba(24,243,255,0.08)',
                    boxShadow: chatOpen ? '0 0 18px rgba(24,243,255,0.06)' : '0 0 30px rgba(255,45,149,0.04)',
                  }}
                >
                  <VideoTrack
                    width={videoWidth}
                    height={videoHeight}
                    trackRef={agentVideoTrack}
                    className={cn(
                      'h-full w-full object-cover opacity-90',
                      chatOpen ? 'scale-110' : 'scale-100'
                    )}
                  />
                </MotionContainer>
              )}
            </AnimatePresence>
          </div>

          <div
            className={cn([
              'grid transition-all duration-500',
              chatOpen && classNames.secondTileChatOpen,
              !chatOpen && classNames.secondTileChatClosed,
            ])}
          >
            {/* Camera & Screen Share Tile */}
            <AnimatePresence>
              {(cameraTrack && isCameraEnabled) || (screenShareTrack && isScreenShareEnabled) ? (
                <MotionContainer
                  key="camera"
                  layout="position"
                  layoutId="camera"
                  initial={{ opacity: 0, scale: 0.8, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.8, y: 20 }}
                  transition={( { ...ANIMATION_TRANSITION, delay: animationDelay } as any )}
                  className={cn(
                    'relative overflow-hidden',
                    'shadow-lg shadow-black/40',
                    'border border-neutral-800 bg-neutral-900',
                    'h-[60px] w-[60px] rounded-lg'
                  )}
                  style={{
                    boxShadow: '0 6px 20px rgba(24,243,255,0.06), inset 0 -4px 12px rgba(255,45,149,0.03)',
                  }}
                >
                  <VideoTrack
                    trackRef={cameraTrack || screenShareTrack}
                    width={(cameraTrack || screenShareTrack)?.publication.dimensions?.width ?? 0}
                    height={(cameraTrack || screenShareTrack)?.publication.dimensions?.height ?? 0}
                    className="h-full w-full object-cover grayscale-[0.1]"
                  />
                  <div className="absolute bottom-1 right-1 h-1.5 w-1.5 rounded-full bg-[#18F3FF] shadow-[0_0_6px_rgba(24,243,255,0.9)]" />
                </MotionContainer>
              ) : null}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
