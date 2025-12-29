# FightingOverlay Quickstart

1. Download **FightingOverlayBootstrap.exe** from GitHub Releases.
2. Double-click it to install/update and launch the **Control Center**.
3. Use the **FightingOverlay** desktop shortcut going forward.

The application installs to `%LOCALAPPDATA%\FightingOverlay\` and keeps only the latest and previous versions.

## Codex Fix Packets (Control Center)

The Control Center can generate “Codex Fix Packets” to help troubleshoot tracking and judge issues.

**Where packets live**
- Pending packets: `%LOCALAPPDATA%\FightingOverlay\data\pending_codex_packets\`
- Sent packets: `%LOCALAPPDATA%\FightingOverlay\data\sent_codex_packets\`
- The combined prompt text is also saved to `%LOCALAPPDATA%\FightingOverlay\data\outputs\`.

**Buttons**
- **Copy Low Pose Warning + Summary**: Copies the low-pose warning string plus the most recent *Evaluation Summary* block.
- **Generate Codex Fix Packet**: Gathers the latest evaluation summary, evaluation JSON paths, last stack trace, run config, and debug pack status into clipboard-ready text (also saved to a `.txt` file).
- **Generate Plan from Pending + Send**: Combines all pending packets into a single prompt asking Codex for ranked root-cause hypotheses, a patch plan, and tests. After generating the prompt, pending packets are moved into the `sent_codex_packets` folder.

**Pending vs sent logic**
- A packet is **pending** if it is newly generated after a run and has not been bundled into a Codex prompt.
- When you click **Generate Plan from Pending + Send**, all pending packets are moved to **sent**.
