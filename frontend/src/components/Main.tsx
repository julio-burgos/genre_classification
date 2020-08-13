import { Component } from "react";
import { withStyles, Container, CssBaseline } from "@material-ui/core";
import React from "react";
const styles: any = (_) => ({});
class Main extends Component<any> {
  render() {
    return (
      <React.Fragment>
        <CssBaseline />
        <Container maxWidth="md" fixed>
          <main>{this.props.children}</main>
        </Container>
      </React.Fragment>
    );
  }
}

export default withStyles(styles)(Main);
