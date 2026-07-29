import {
  createContext,
  useContext,
  useState
} from "react";

const TranslationContext =
  createContext();

export function TranslationProvider({
  children
}) {

  const [history, setHistory] =
    useState([]);

  function addTranslation(item) {

    setHistory((prev) => [
      item,
      ...prev
    ]);

  }

  function clearHistory() {

    setHistory([]);

  }

  return (

    <TranslationContext.Provider
      value={{
        history,
        addTranslation,
        clearHistory
      }}
    >

      {children}

    </TranslationContext.Provider>

  );
}

export function useTranslation() {

  return useContext(
    TranslationContext
  );

}