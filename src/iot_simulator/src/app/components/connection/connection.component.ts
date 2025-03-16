import { CommonModule } from "@angular/common";
import { Component, Input } from "@angular/core";
import { Connection, TransmittingStatus } from "@models/connection";
import { Position } from "@models/position";

@Component({
    selector: "app-connection",
    imports: [CommonModule],
    templateUrl: "connection.component.html",
    styleUrl: "connection.component.css",
})
export class ConnectionComponent {
    @Input({ required: true })
    public connection!: Connection;
    protected get transmitting(): boolean {
        return this.connection.transmitting !== TransmittingStatus.NONE;
    }
    protected get reverse(): boolean {
        return (
            this.connection.transmitting === TransmittingStatus.NODE_TO_ROUTER
        );
    }
    protected get router(): Position {
        return this.connection.router.position;
    }
    protected get node(): Position {
        return this.connection.node.position;
    }
    protected get w(): number {
        return Math.abs(this.node.x - this.router.x) + 60;
    }
    protected get h(): number {
        return Math.abs(this.node.y - this.router.y) + 60;
    }
    protected get t(): Position {
        return {
            x: (this.router.x + this.node.x) / 2 - this.w / 2,
            y: (this.router.y + this.node.y) / 2 + this.h / 2,
        };
    }
}
