import React, { Fragment } from "react";
import MyRoutes from "./components/MyRoutes";
import Main from "./components/Main";
import MenuBar from "./components/MenuBar";

const App: React.FC = () => {
  return (
    <Fragment>
      <MenuBar />
      <Main>
        <MyRoutes />
      </Main>
    </Fragment>
  );
};

export default App;
