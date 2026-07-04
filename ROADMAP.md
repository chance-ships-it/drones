# Roadmap: from this simulator toward a real autonomous drone

This is a passion-project map, ordered so each stage builds on the last and
each one *works* before you move on. You don't need all of it — stop wherever
it stops being fun. Rough difficulty in brackets.

## Where you are now

A rigid-body quadrotor simulator that trains hover and waypoint policies with
PPO, entirely on a MacBook. That is genuinely the hard 60% — you have a working
sim-and-learn loop. Everything below makes it *more real* and *more capable*.

## Stage 1 — Get fluent with the loop you have  [easy]

Before adding anything, get comfortable changing what's here. Good first edits:

- **Tune the reward** (`drone_sim/env.py`, `_reward`). Change a coefficient,
  retrain, watch the GIF. This is the single most important skill in RL — most
  of the work is reward design, not algorithms.
- **Fix the waypoint crashes** (see the known-limitation note in the README).
  Try: a larger spin penalty, a curriculum that starts targets close and moves
  them out as success rises, or observation normalization.
- **Add a metric** to the training log (e.g. crash rate) so you can *measure*
  whether a change helped instead of guessing.

Run `python tests/test_physics.py` after each change so you know you didn't
break the physics.

## Stage 2 — Make the simulator harder to fool  [medium]

A policy that only works in a clean sim won't survive contact with reality.
Add the things reality has that your sim doesn't:

- **Sensor noise + latency.** Add Gaussian noise to the observation in `_obs()`
  and delay the action by a step or two. Real drones never see perfect state.
- **Wind and disturbances.** Add a random force in `QuadrotorPhysics.step` — a
  steady breeze plus gusts. Train with it and the policy learns to reject it.
- **Wider domain randomization.** Push the ranges in `QuadrotorParams`. If the
  policy survives a *distribution* of drones, one real drone is just a sample.
- **An asymmetric critic** (advanced): the critic sees true state, the actor
  sees only noisy sensors. This is the trick behind champion-level drone racing
  policies (see the "Reaching the limit" / Swift papers).

## Stage 3 — New capabilities  [medium]

Each of these is mostly a `_sample_targets` + `_reward` change — the same shape
as the hover→waypoint jump you've already seen work:

- **Velocity / trajectory tracking:** follow a moving reference, not a point.
- **Racing through gates:** oriented gates instead of spheres; reward passing
  through in the right direction. This is a real research benchmark.
- **Perching / precision landing:** reward touching a pad softly and level.
- **Recovery:** start upside-down or tumbling and reward getting upright.

## Stage 4 — Bridge to hardware, in software first  [medium-hard]

You do *not* need a drone yet. De-risk on your laptop:

- **Export the policy.** The actor is a tiny 2×128 MLP. Save it to ONNX and
  confirm you can run inference outside PyTorch — that's what runs onboard.
- **Match a real airframe.** Pick a small drone (see Stage 5) and set
  `QuadrotorParams` to *its* measured mass, arm length, and thrust. Now the sim
  is a digital twin of a specific aircraft.
- **Cross-check against a reference sim.** Fly the same commands in a
  battle-tested simulator (Betaflight SITL, jMAVSim, or Gazebo/PX4) and compare
  trajectories. If they diverge badly, your physics is missing something.

## Stage 5 — Real hardware  [hard, costs money]

The well-trodden hobbyist path for RL-on-drones:

- **Airframe:** a small 3–5" quadrotor, or a Crazyflie 2.1 (tiny, indoor-safe,
  great for research — this is what most academic RL-drone work uses).
- **Flight stack:** PX4 or Betaflight. Your policy outputs thrust/rate commands
  that the flight controller executes.
- **State estimation:** indoors, a motion-capture rig (OptiTrack/Vicon) or a
  Crazyflie Lighthouse deck gives you position; outdoors, GPS + IMU fusion.
- **Deploy:** run your exported policy on a companion computer (or the
  Crazyflie itself for small nets), feeding it estimated state, streaming out
  motor commands. Start with a safety tether and a kill switch.

Expect the *sim-to-real gap*: the policy will fly worse in reality than in sim.
Closing that gap is the whole game — it's what Stages 2 and 4 are insurance for.

## Things worth reading

- Google/ETH "Champion-level drone racing with deep RL" (Swift, Nature 2023)
- "Learning to Fly" / Crazyflie RL papers for the small-drone hobbyist path
- OpenAI "Sim-to-real" domain-randomization write-ups (the dexterity work)
- Spinning Up in Deep RL (OpenAI) — the best free intro to the algorithms

## How to work through this

Pick one bullet. Branch (`git checkout -b wind`), make the change, run the
tests, retrain, watch the GIF, commit if it's better, throw it away if it's
not. That branch-experiment-measure loop is the entire craft. Have fun.
