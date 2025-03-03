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
        return this.connection.transmitting === TransmittingStatus.DEVICE_TO_ROUTER;
    }
    protected get router(): Position {
        return this.connection.router.position;
    }
    protected get device(): Position {
        return this.connection.device.position;
    }
    protected get w(): number {
        return Math.abs(this.device.x - this.router.x) + 60;
    }
    protected get h(): number {
        return Math.abs(this.device.y - this.router.y) + 60;
    }
    protected get t(): Position {
        return {
            x: (this.router.x + this.device.x) / 2 - this.w / 2,
            y: (this.router.y + this.device.y) / 2 + this.h / 2,
        };
    }
}
