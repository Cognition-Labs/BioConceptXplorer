import { RecoilRoot, useRecoilValue } from "recoil";
import { ChakraProvider, Text } from "@chakra-ui/react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import theme from "./theme";
import HomePage from "./pages/HomePage";
import ExplorePage from "./pages/ExplorePage";

import { ColorModeScript } from "@chakra-ui/react";
import Navbar from "./components/Navbar";
import { NAV_ITEMS } from "./constants";

import { loadingState } from "./atoms";
import { useEffect } from "react";

function App() {
  const loading = useRecoilValue(loadingState);

  // this runs once when the app is first loaded, heartbeat to backend to make sure it's up
  useEffect(() => {
    console.log("setup heartbeat")
    setInterval(() => {
      console.log("heartbeat")
      fetch("https://shreyj1729--bioconceptvec-heartbeat.modal.run", {
        method: "GET",
      })
    }, 1000 * 5)
  }, []);
  return (
    <>
      <ColorModeScript initialColorMode="dark" />
      <ChakraProvider theme={theme}>
        <BrowserRouter>
          <Navbar navItems={NAV_ITEMS} loading={loading} />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/explore" element={<ExplorePage />} />
            <Route path="*" element={<Text pt="10vh">404</Text>} />
          </Routes>
        </BrowserRouter>
      </ChakraProvider>
    </>
  );
}

export default App;
