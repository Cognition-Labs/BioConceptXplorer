import { atom, selector } from "recoil";

export const loadingState = atom({
  key: "loadingState",
  default: false,
});

export const expressionAtom = atom({
  key: "expressionState",
  default: [""],
});

export const expressionResult = atom({
  key: "expressionResult",
  default: "",
});
