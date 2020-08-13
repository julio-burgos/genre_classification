import React from "react"
import { Switch, Route } from "react-router-dom";

import { routes } from "../routes";
import GenreClassification from "./GenreClassification";
import Recomender from "./Recomend";


const MyRoutes = (props) => {
    console.log(props);
    
   return (
        <React.Fragment>
            <Switch>
                <Route key = {0}  exact   path= "/"  component = {GenreClassification} />             
                <Route key = {1}  exact  path =  "/recomendsong" component = {Recomender} />             
            </Switch>
        </React.Fragment>

    );
}

export default MyRoutes
