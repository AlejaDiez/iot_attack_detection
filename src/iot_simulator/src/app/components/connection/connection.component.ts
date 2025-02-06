import { Component, Input } from "@angular/core";
import { Connection } from "../../models/connection";
import { Position } from "../../models/position";

@Component({
    selector: "app-connection",
    imports: [],
    templateUrl: "./connection.component.html",
    styleUrl: "./connection.component.css",
})
export class ConnectionComponent {
    @Input()
    public connection!: Connection;

    public get origin(): Position {
        return this.connection.origin.position;
    }

    public get destination(): Position {
        return this.connection.destination.position;
    }

    public get w(): number {
        return Math.abs(this.destination.x - this.origin.x) + 60;
    }

    public get h(): number {
        return Math.abs(this.destination.y - this.origin.y) + 60;
    }

    public get t(): Position {
        return {
            x: (this.origin.x + this.destination.x) / 2 - this.w / 2,
            y: (this.origin.y + this.destination.y) / 2 + this.h / 2,
        };
    }
}
