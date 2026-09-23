# TODOs

## UX & Gameplay

- **Make gameplay fun** — The interface works but lacks engaging game mechanics or feedback. Refine rewards, pacing, and player feedback to make the experience enjoyable. The floor is currently ~3 s per move (1.5 s player turn + ~1 s hosted prediction + 0.5 s reveal); predicting speculatively during the player's turn would cut it but costs quota and breaks the "ghost rows show the board TabPFN actually saw" promise.

- **Mobile-friendly UI** — Design and test a responsive layout so the game is playable on phones and tablets, not just desktop. The board overlay and arrow view already scale with the canvas; the two-panel layout, the table, and touch targets for apple moves do not.
