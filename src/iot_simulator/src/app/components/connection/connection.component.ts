import { AfterViewInit, Component, Input, OnInit } from "@angular/core";
import { NodeComponent } from "../node/node.component";

@Component({
    selector: "app-connection",
    imports: [],
    templateUrl: "./connection.component.html",
    styleUrl: "./connection.component.css",
})
export class ConnectionComponent {
    @Input()
    origin!: NodeComponent;
    @Input()
    destination!: NodeComponent;

    get w(): number {
        return Math.abs(this.destination.x - this.origin.x) + 60;
    }
    get h(): number {
        return Math.abs(this.destination.y - this.origin.y) + 60;
    }
    get tx(): number {
        return (this.origin.x + this.destination.x) / 2 - this.w / 2;
    }
    get ty(): number {
        return (this.origin.y + this.destination.y) / 2 + this.h / 2;
    }
}
