# Prediction math

This document matches the implementation in `calibration.py`,
`trajectory.py`, `tof.py`, `engine.py`, and `servo.py`.

## 1. Pixel coordinates to metres

Four non-collinear points define a projective homography \(H\). For a detected
pixel centre \((u, v)\):

\[
\begin{bmatrix}
\tilde{x} \\
\tilde{y} \\
\tilde{w}
\end{bmatrix}
=
H
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix},
\qquad
x = \frac{\tilde{x}}{\tilde{w}},
\quad
y = \frac{\tilde{y}}{\tilde{w}}.
\]

World \(x\) runs along the floor, and \(y\) is height above the floor. OpenCV
estimates \(H\); calibration stores both directions and the reprojection RMSE.

This is a planar model. It is accurate only while the ball stays near the
calibrated side-on throwing plane.

## 2. ToF passage trigger

Let \(d_i\) be a valid range sample. After at least ten background samples:

\[
b = \operatorname{median}(d_{i-N+1}, \ldots, d_i).
\]

The gate triggers when:

\[
b - d_i \geq 80\text{ mm}.
\]

It re-arms when the drop returns to at most 35 mm. The trigger is considered
recent for 3 seconds. A trigger clears all previous camera observations.

The ToF measurement does not enter the trajectory equations; it supplies event
timing and rejects unrelated colour detections.

## 3. Ballistic observation model

Choose the latest observation time as \(t=0\). For earlier samples,
\(\tau_i = t_i - t_\text{latest}\), so \(\tau_i \leq 0\).

Horizontal motion:

\[
x_i = x_0 + v_x \tau_i + \epsilon_{x,i}.
\]

Vertical motion with fixed gravitational acceleration:

\[
y_i = y_0 + v_y \tau_i - \frac{1}{2}g\tau_i^2 + \epsilon_{y,i},
\qquad
g = 9.80665\text{ m/s}^2.
\]

Fixing gravity leaves two linear unknowns per axis and stabilises early webcam
predictions. At least six observations spanning at least 0.12 seconds are
required.

## 4. Robust weighted fit

Each detection has quality \(q_i \in [0.05, 1]\). The implementation performs
eight iteratively reweighted least-squares passes. For residual \(r_i\) and
Huber threshold \(\delta=0.025\) m:

\[
w_i =
q_i
\begin{cases}
1, & |r_i| \leq \delta \\
\delta / |r_i|, & |r_i| > \delta.
\end{cases}
\]

The code applies this weight independently to horizontal and vertical
residuals. A large isolated centroid error therefore has limited influence.

The combined fit error is:

\[
\operatorname{RMSE} =
\sqrt{
\frac{\sum_i q_i(r_{x,i}^2+r_{y,i}^2)}
{\sum_i q_i}
}.
\]

## 5. Future floor intersection

At the newest observation, solve for future \(t>0\) where \(y(t)=y_f\), with
floor height \(y_f=0\):

\[
\frac{1}{2}a_y t^2 + v_y t + (y_0-y_f)=0,
\qquad a_y=-g.
\]

The quadratic roots are:

\[
t =
\frac{-v_y \pm
\sqrt{v_y^2-4(0.5a_y)(y_0-y_f)}}
{a_y}.
\]

PREACT chooses the smallest positive root. It rejects intersections less than
40 ms or more than 3 s in the future.

The predicted landing coordinate is:

\[
x_\text{land}=x_0+v_x t_\text{impact}.
\]

Fits are also rejected if horizontal or vertical speed exceeds 15 m/s or the
latest observation is materially below the calibrated floor.

## 6. Confidence

Six bounded factors are calculated:

\[
s_n=\min(1,n/14),
\quad
s_T=\min(1,T_\text{observed}/0.35),
\]

\[
s_r=e^{-\operatorname{RMSE}/0.04},
\quad
s_i=\text{inlier fraction},
\quad
s_q=\text{mean detection quality},
\]

\[
s_h=e^{-0.35\max(0,t_\text{impact}-T_\text{observed})}.
\]

The final score is their geometric mean:

\[
C=(s_n s_T s_r s_i s_q s_h)^{1/6}.
\]

The servo planner requires \(C \geq 0.45\) by default. A geometric mean ensures
one very weak dimension meaningfully reduces confidence.

## 7. Current or future point to servo angle

The target is the current ball point \((x,y)\) in tracking mode and the future
floor point \((x_\text{land},0)\) in landing mode. Let the servo axis be at
\((x_s,y_s)\) in the same calibrated vertical plane. The target bearing is:

\[
\theta=\operatorname{atan2}(y-y_s,\ x-x_s).
\]

For servo zero angle \(a_0\) and direction \(d\in\{-1,1\}\):

\[
a=a_0+d\,\theta\frac{180}{\pi}.
\]

The result is clamped to the configured servo endpoint envelope. This
`planar-pointing` mapping is required if the same one-axis arm must point both
upward at an airborne ball and downward at its future floor contact. It assumes
the throw stays close to the calibrated plane.

For a compact landing-only scale, the alternative `linear` mapping remains:

For calibrated landing limits \(x_\min,x_\max\):

\[
f =
\operatorname{clip}
\left(
\frac{x_\text{land}-x_\min}{x_\max-x_\min},
0,1
\right).
\]

For endpoint angles \(a_\min,a_\max\):

\[
a=a_\min+f(a_\max-a_\min).
\]

The laptop clamps to its calibrated endpoint envelope and 0–180°. Firmware then
applies a narrower default 10–170° mechanical limit and converts the angle to a
600–2400 µs pulse at 50 Hz.

Tracking additionally requires detection quality of at least 0.35 by default.
The arm returns to neutral when no accepted detection has been available for
350 ms. Landing mode retains the independent confidence and ToF gates above.

## 8. Latency target

At 30 FPS, six samples span roughly 0.17 seconds. A typical controlled throw
lasting 0.8–1.2 seconds therefore leaves several hundred milliseconds for a
stable pre-impact prediction. The acceptance target is:

- first stable output at least 300 ms before impact;
- landing error at most 0.20 m in at least 8 of 10 throws.

JSONL telemetry records observation count, fit error, confidence, time to
impact, ToF state, and servo target for every processed frame.
