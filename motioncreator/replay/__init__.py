"""Replay of compiled motions on a real G1 under Unitree's default policy (arms over rt/arm_sdk).

Ported from g1-live-studio (g1live/arm_sdk_writer.py, dds_reader.py, replay.py at 8411a3d), with the replay procedure of
robot-studio (dashboard/runner.py). Nothing here imports the Unitree SDK except the two child-process scripts.
"""
