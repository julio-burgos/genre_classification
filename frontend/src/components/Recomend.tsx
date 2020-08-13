import React, { Component } from "react";
import TextField from "@material-ui/core/TextField";
import Autocomplete from "@material-ui/lab/Autocomplete";
import {
  CircularProgress,
  Button,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
  makeStyles,
  withStyles
} from "@material-ui/core";
import DeleteIcon from "@material-ui/icons/Delete";
import CheckBoxOutlineBlankIcon from "@material-ui/icons/CheckBoxOutlineBlank";
import CheckBoxIcon from "@material-ui/icons/CheckBox";

import Checkbox from "@material-ui/core/Checkbox";

const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
const checkedIcon = <CheckBoxIcon fontSize="small" />;
const styles = theme => ({
  root: {
    width: '100%',
    marginTop:"1rem",
    backgroundColor: theme.palette.background.paper,
  },
});

class Recomender extends Component<any, any> {
  /**
   *
   */

  constructor(params) {
    super(params);
    this.state = {
      selectedsongs: [],
      open: false,
      options: [],
      loading: true,
      recomendedSongs: []
    } as any;
  }
  setOptions = val => {
    const open = this.state.open;
    this.setState({ options: val.slice(), loading: open && val.length === 0 });
  };
  setOpen = val => {
    this.setState({ open: val });
  };

  setselectsong = (option, sel) => {
    if (!sel.selected) {
      const songs = this.state.selectedsongs.slice();
      songs.push(option);
      this.setState({ selectedsongs: [...songs] });
    }
  };

  handleChange = async ev => {
    this.setOpen(false);
    console.log(ev.target.value);
    const q = ev.target.value as string;
    if (q == null || q.trim() == "") {
      this.setOptions([]);
    }

    const response = await fetch("/searchsongs?q=" + q, {
      
      method: "GET"
    });

    const songs = await response.json();
    console.log(songs);
    this.setOptions(songs);
    this.setOpen(true);
  };
  handleSelectedSong = option => {
    console.log(option.target.value);
    
  };

  handleClick = async ev => {
    const response = await fetch("/getRecommendations", {
      
      method: "POST",
      body: JSON.stringify({
        selectedsongs: this.state.selectedsongs.map(x => x)
      }),
      headers: new Headers({
        "Content-Type": "application/json",
        Accept: "application/json"
      })
    });
    const songs = await response.json();
    console.log(songs);

    this.setState({
      recomendedSongs: songs.slice()
    });
  };
  render() {
    const { classes } = this.props;
    return (
      <React.Fragment>
        <Autocomplete
          disableCloseOnSelect
          multiple
          id="song-selector"
          style={{ marginTop: "2rem" }}
          open={this.state.open}
          clearOnEscape
          onOpen={() => {
            this.setOpen(true);
          }}
          onClose={() => {
            this.setOpen(false);
          }}
          getOptionLabel={option => option.name}
          onChange={this.handleSelectedSong}
          getOptionSelected={(option, value) => option.id === value.id}
          renderOption={(option, sel) => (
            <span
              onClick={ev => this.setselectsong(option, sel)}
              style={{ width: "100%" }}
            >
              <Checkbox
                icon={icon}
                checkedIcon={checkedIcon}
                style={{ marginRight: 8 }}
                checked={sel.selected}
              />
              {option.name + " by " + option.artist}
            </span>
          )}
          options={this.state.options}
          loading={this.state.loading}
          freeSolo
          disableClearable
          autoSelect={true}
          renderInput={params => (
            <React.Fragment>
              <TextField
                {...params}
                onChange={this.handleChange}
                label="Select Track"
                fullWidth
                variant="outlined"
                InputProps={{
                  ...params.InputProps,
                  endAdornment: (
                    <React.Fragment>
                      {this.state.loading ? (
                        <CircularProgress color="inherit" size={20} />
                      ) : null}
                      {params.InputProps.endAdornment}
                    </React.Fragment>
                  )
                }}
              />
            </React.Fragment>
          )}
        />

        <Button
          color="primary"
          variant="outlined"
          onClick={this.handleClick}
          style={{
            marginTop: "-3.5em",
            marginRight: "-9em",
            float: "right"
          }}
        >
          Recommend
        </Button>
        { (
          <List component="nav" className={classes.root}>
            {this.state.recomendedSongs.map((song,index) => (
            <ListItem key={index}>
              <a target="_blank" href={song.track_url}>
              <ListItemText
                primary={song.track_title}
                secondary={song.artist_name}
              />
              </a>
            </ListItem>
            )) }
          </List>
        )}
      </React.Fragment>
    );
  }
}

export default  withStyles(styles)(Recomender)