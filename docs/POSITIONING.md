# Positioning

> **PREACT calls its shot.** The ball leaves your hand, and before it is
> halfway down the system has committed to a landing spot, out loud, on
> hardware. Then the ball lands on it.

## The one insight

Interception is a race between prediction and arrival. Every defensive
system runs the same loop: sense the object, fit a motion model, predict
where it will be, and commit an effector there early enough that the
effector's own latency stops mattering.

Historically the expensive parts of that loop were the sensor (radar)
and the effector (a guided missile). The prediction layer in the middle
is just math, and the math does not care whether the observations come
from a million-dollar radar or a commodity camera. It only needs a
correct motion model and enough early, noisy points.

## Why the demo is cost-efficient interception

A guided interceptor is expensive because it steers itself in flight. It
steers because the system is not sure where the target will be, so it
buys a projectile smart enough to correct mid-course, and throws it away
every shot. **Guidance is uncertainty compensation, purchased per shot.**

Better, earlier prediction shrinks that uncertainty. Once the predicted
position is tighter than a dumb effector's reach, the self-correcting
missile can be replaced by an unguided round, a net, a jammer, or a
second of laser time. This is already how C-RAM works: Phalanx fires
plain 20 mm rounds and hits incoming mortars because a radar and a
prediction computer aim them. The intelligence stays on the ground and
is reused every engagement; the thing that flies is cheap.

What remains expensive in that ground stack is the sensing and
fire-control (a C-RAM battery is roughly $10–20 M). PREACT runs the same
sense–predict–commit loop on a laptop camera and a microcontroller.
Cheap sensing does not only lower the price of one system; it lets the
capability exist where a large radar never could: a checkpoint, a
vehicle, a warehouse roof.

**We do not make missiles cheaper. We make the aim good enough that you
stop needing missiles.**

## The pointer is the argument

The servo pointer has no guidance. After the lock signal it commits to a
spot and nothing can correct it. When the ball lands on that spot anyway,
that is a dumb effector hitting because prediction did all the work,
which is the entire cost story shown physically.

## The output is already an effector interface

At lock, PREACT emits a target position and a commit trigger. That is a
fire-control message. A turret, a pan/tilt gun mount, or an interceptor
drone's autopilot all take the same two things: **where** and **when**.
The servo is simply the cheapest effector we could plug into that port;
swapping it for a larger effector is swapping what listens on the line,
not rewriting the system. We are the targeting layer, not the projectile.

## Threat scope, stated honestly

- **Unguided threats** — mortars, artillery, unguided rockets, and
  grenades dropped from drones the moment they release — are ballistic
  and follow exactly the physics this demo predicts. This class is most
  of what actually gets intercepted.
- **Guided threats** need a bank of motion models (straight, turn, dive)
  and continuous re-commitment over a short horizon rather than one
  fixed-gravity fit. The prediction horizon only has to cover the
  effector's latency, which we already clear. In practice the first move
  against a guided drone is to jam its link, which turns it back into an
  unguided object, our problem class again.
- **Point defense, not area defense.** Dumb effectors have short reach,
  so this is a site-protection story. That is precisely the gap that
  cheap drones and mortars currently exploit.

## Dual-use bridge

The same predict-before-impact loop is direction-agnostic. It applies
unchanged to industrial safety (stop a machine before a part is thrown
into a walkway), robotics and logistics (catch or divert moving items),
and sports and training (call the landing of a served or thrown ball).
The defensive-interception framing is the sharpest edge of one
capability: **early motion prediction on commodity hardware, with an
inspectable confidence gate before anything actuates.**
