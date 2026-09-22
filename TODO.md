# TODOs

## UX & Gameplay

- **Make gameplay fun** — Currently the interface works but lacks engaging game mechanics or feedback. Refine rewards, pacing, and player feedback to make the experience enjoyable.

- **Mobile-friendly UI** — Design and test responsive layout so the game is playable on phones and tablets, not just desktop.

## Technical Issues

- **Prediction graphics broken** — The feature showing snake predictions on different board cells is not working correctly. Either fix the implementation or remove this visualization entirely.

## Flow & Onboarding

- **Redesign "Play 1000 random steps" flow** — The current button is confusing when the goal is to play against TabPFN. Instead:
  - Introduce this step as part of guided onboarding that explains the gameplay loop
  - Clearly communicate that TabPFN needs to practice with random games before it can play competitively
  - Load the replay table silently during this practice phase so the heavy I/O is hidden from the user
  - Make this feel like a necessary setup phase ("Preparing TabPFN...") rather than a standalone action
