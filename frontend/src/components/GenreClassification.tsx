import React, { Component } from "react";
import {
  Typography,
  Grid,
  makeStyles,
  FormControl,
  InputLabel,
  Input,
  FormHelperText,
  Button,
  Select,
  MenuItem,
  IconButton
} from "@material-ui/core";
import { HorizontalBar } from "react-chartjs-2";

import CloudUploadOutlinedIcon from "@material-ui/icons/CloudUploadOutlined";
import { CircularProgress } from "@material-ui/core";
import { getmonofromfile } from "../utils/spectralanalysis";
const useStyles = makeStyles(theme => ({
  content: {
    padding: theme.spacing(2),
    textAlign: "center",
    color: theme.palette.text.primary
  },
  uploadbutton: {
    marginTop: theme.spacing(1)
  },
  selectmodel: {
    minWidth: "10em"
  },
  circularprogress: {
    color: "#428EC6",
    marginTop: theme.spacing(12)
  }
}));

export default function GenreClassification() {
  const classes = useStyles();
 
  const [modelSelected, setModel] = React.useState("");
  const [isLoaded, setLoaded] = React.useState(false);

  const [data, setData] = React.useState();

  const options = {
    scales: {
      xAxes: [
        {
          ticks: {
            suggestedMin: 0,
            suggestedMax: 100
          }
        }
      ]
    },
    legend: false,
    maintainAspectRatio: true
  };
  
  const handleClick = async function(event:any) {
    setLoaded(false);
    const fileInput = document.getElementById("upload") as any;
    const filename = fileInput.files[0].name;
    const mono = (await getmonofromfile(event)).arraySync();
    const rawres= await fetch("/getgenre",{
      
      method: "POST",
      body: JSON.stringify({
        mono
      }),
      headers: new Headers({
        "Content-Type": "application/json",
        Accept: "application/json"
      })
    })
    const json = await rawres.json();
    console.log(json);
    
    const predictions = json.predictions.map(v => v*100)
    console.log(json.predictions);
    console.log(predictions);
    

    setData(
      {
        labels: ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae','rock'],
        datasets: [
          {
            label: filename,
            backgroundColor: "#ff6c850f",
            borderColor: "#ff6c85ff",
            borderWidth: 1,
            hoverBackgroundColor: "#ff6c852f",
            hoverBorderColor: "#ff6c85ff",
            data: predictions,
            maxBarThickness: 40
          }
        ]
      }
    );
    setLoaded(true);
  };
  const handleChange = function(event) {
    setModel(event.target.value);
  };

  return (
    <Grid container className={classes.content}>
      <Grid item xs={12}>
        <Typography variant="h4">Genre Classification</Typography>
      </Grid>

      
      <Grid item xs={12}>
        <Button
          variant="contained"
          component="label"
          className={classes.uploadbutton}
        >
          Upload Song
          <CloudUploadOutlinedIcon />
          <input
            type="file"
            id="upload"
            style={{ display: "none" }}
            onChange={handleClick}
          />
        </Button>
      </Grid>
      <Grid item xs={12}>
        {isLoaded ? (
          <HorizontalBar
            data={data}
            width={100}
            height={40}
            options={options}
          />
        ) : (
          <CircularProgress
            className={classes.circularprogress}
            size={150}
            thickness={3}
          />
        )}
      </Grid>
    </Grid>
  );
}
