import { Component } from "@angular/core";
import { CanvasComponent } from "./components/canvas/canvas.component";
import { NodeComponent } from "./components/node/node.component";
import { ConnectionComponent } from "./components/connection/connection.component";

@Component({
    selector: "app-root",
    templateUrl: "app.component.html",
    imports: [CanvasComponent, NodeComponent, ConnectionComponent],
})
export class AppComponent {}
