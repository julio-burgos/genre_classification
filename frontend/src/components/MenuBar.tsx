import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Tabs from "@material-ui/core/Tabs";
import Tab from "@material-ui/core/Tab";
import AlbumIcon from "@material-ui/icons/Album";
import { Link } from "react-router-dom";
import { routes } from "../routes";
import {
  Toolbar,
  Button,
  Typography,
  IconButton,
  Avatar
} from "@material-ui/core";
import LibraryMusicIcon from "@material-ui/icons/LibraryMusic";
import AccountCircleIcon from "@material-ui/icons/AccountCircle";
import UPCLogo from "../images/upc-logo.svg";
const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
    backgroundColor: theme.palette.background.paper
  },
  toolbarButtons: {
    marginLeft: "auto"
  },
  indicator: {
    backgroundColor: "#428EC6"
  },
  logo:{
    width: 50,
    height: 50,
  }
}));

export default function MenuBar() {
  const classes = useStyles();
  const [value, setValue] = React.useState(0);
  const handleChange = (event, newValue) => {
    setValue(newValue);
    console.log(newValue);
  };
  const icons = [<AlbumIcon />, <LibraryMusicIcon />];
  const tabs = routes.map((route, index) => (
    <Tab
      icon={icons[index]}
      label={route.label}
      component={Link}
      value={index}
      key={index}
      to={route.path}
    />
  ));
  console.log(tabs);

  return (
    <div className={classes.root}>
      <AppBar position="static" className={classes.indicator}>
        <Toolbar>
          <Avatar className={classes.logo} alt="UPCLogo" src={UPCLogo} />
       
        </Toolbar>
      </AppBar>
      <Tabs
        value={value}
        aria-label="simple tabs example"
        centered
        onChange={handleChange}
      >
        {tabs}
      </Tabs>
    </div>
  );
}
