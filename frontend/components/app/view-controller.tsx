'use client';

import { useRef } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useRoomContext } from '@livekit/components-react';
import { useSession } from '@/components/app/session-provider';
import { SessionView } from '@/components/app/session-view';
import { WelcomeView } from '@/components/app/welcome-view';

const MotionWelcomeView = motion.create(WelcomeView);
const MotionSessionView = motion.create(SessionView);

// loosen types to avoid motion/react typing mismatch
const VIEW_MOTION_PROPS: any = {
  variants: {
    visible: {
      opacity: 1,
    },
    hidden: {
      opacity: 0,
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.5,
    ease: 'linear',
  },
};

export function ViewController() {
  const room = useRoomContext();
  const isSessionActiveRef = useRef(false);
  const { appConfig, isSessionActive, startSession } = useSession();

  // keep ref in sync for animation-complete cleanup
  isSessionActiveRef.current = isSessionActive;

  const handleAnimationComplete = () => {
    // disconnect the room after the exit animation finishes to avoid abrupt audio cut
    if (!isSessionActiveRef.current && room.state !== 'disconnected') {
      room.disconnect();
    }
  };

  return (
    <AnimatePresence mode="wait">
      {!isSessionActive && (
        <MotionWelcomeView
          key="welcome"
          {...(VIEW_MOTION_PROPS as any)}
          startButtonText={appConfig.startButtonText}
          onStartCall={startSession}
        />
      )}

      {isSessionActive && (
        <MotionSessionView
          key="session-view"
          {...(VIEW_MOTION_PROPS as any)}
          appConfig={appConfig}
          onAnimationComplete={handleAnimationComplete as any}
        />
      )}
    </AnimatePresence>
  );
}
