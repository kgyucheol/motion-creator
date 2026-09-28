# Decoupled WBC workbench

`/decoupled-wbc` is a separate simulation page that combines a Motion Creator
reference with NVIDIA's decoupled whole-body controller.

- The authored left/right arm joints (14 DoF) are tracked with direct PD control.
- The authored waist pose becomes the controller's torso RPY command. The policy
  continues to own the physical waist and both legs (15 DoF).
- `W/S`, `A/D`, and `Q/E` change forward, lateral, and yaw velocity. `1/2`
  changes body height and `Z` clears all teleoperation commands.
- Play starts both simulation and recording. Stop pauses and clears locomotion
  velocity. Reset restores the initial simulation and clears the recording.
- `오브젝트 콜리전` hides imported scene meshes and shows the exact primitive
  proxies used by WBC physics: one longest-axis inscribed cylinder per ramen
  bundle and five box panels for an open carton.
- Saved recordings contain `motion.npz`, `motion.csv`, `commands.csv`, and
  `metadata.json` under `motions/Decoupled_WBC_*`.

## One-time setup

The policy files and MJCF meshes are kept out of git. Copy and verify them from
a GR00T-WholeBodyControl checkout:

```bash
scripts/setup-decoupled-wbc.sh ../GR00T-WholeBodyControl
```

The setup records the upstream revision and verifies the hashes in
`integrations/decoupled-wbc-assets.json`. The application uses the CPU policy
environment (`.conda-policy`) when it is available.
