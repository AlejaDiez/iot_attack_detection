import { Component, Input } from "@angular/core";
import { Connection } from "../../models/connection";

@Component({
    selector: "app-connection",
    imports: [],
    templateUrl: "./connection.component.html",
    styleUrl: "./connection.component.css",
})
export class ConnectionComponent {
    @Input()
    connection!: Connection;

    get w(): number {
        return (
            Math.abs(
                this.connection.destination.position.x -
                    this.connection.origin.position.x
            ) + 60
        );
    }
    get h(): number {
        return (
            Math.abs(
                this.connection.destination.position.y -
                    this.connection.origin.position.y
            ) + 60
        );
    }
    get tx(): number {
        return (
            (this.connection.origin.position.x +
                this.connection.destination.position.x) /
                2 -
            this.w / 2
        );
    }
    get ty(): number {
        return (
            (this.connection.origin.position.y +
                this.connection.destination.position.y) /
                2 +
            this.h / 2
        );
    }
}
