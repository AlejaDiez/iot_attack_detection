import { Component } from "@angular/core";
import { CanvasComponent } from "./components/canvas/canvas.component";
import { NodeComponent } from "./components/node/node.component";
import { ConnectionComponent } from "./components/connection/connection.component";
import { Device, DeviceType } from "./models/device";
import { Router } from "./models/router";
import { CommonModule } from "@angular/common";
import { RouterOutlet } from "@angular/router";

@Component({
    selector: "app-root",
    templateUrl: "app.component.html",
    imports: [
        CommonModule,
        RouterOutlet,
        CanvasComponent,
        NodeComponent,
        ConnectionComponent,
    ],
    styles: [
        `
            .router {
                position: relative;
                z-index: 1;
            }
        `,
    ],
})
export class AppComponent {
    nodes: Device[] = [
        new Router("192.168.0.1", "Router"),
        new Device("192.168.0.31", "RPi4", DeviceType.IOT, [], {
            x: -200,
            y: -100,
        }),
        new Device("192.168.0.24", "Ordenador", DeviceType.COMPUTER, [], {
            x: 100,
            y: 100,
        }),
    ];

    constructor() {
        // Connect devices
        this.nodes[1].connect(this.nodes[0], 1000);
        this.nodes[2].connect(this.nodes[0], 2000);
    }
}
