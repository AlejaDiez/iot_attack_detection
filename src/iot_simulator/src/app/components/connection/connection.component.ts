import { CommonModule } from "@angular/common";
import { Component, inject, input, InputSignal } from "@angular/core";
import { HlmDialogService } from "@components/ui/ui-dialog-helm/src";
import { Connection, ConnectionStatus } from "@models/connection";
import { Position } from "@models/position";
import { EditConnectionDialogComponent } from "@routes/dialogs/edit-connection-dialog.component";
import { NetworkService } from "@services/network.service";

@Component({
    selector: "app-connection",
    imports: [CommonModule],
    templateUrl: "connection.component.html",
    styleUrl: "connection.component.css",
    host: { class: "contents" },
})
export class ConnectionComponent {
    public readonly dialog: HlmDialogService = inject(HlmDialogService);
    public readonly connection: InputSignal<Connection> = input.required();
    protected get animate(): boolean {
        return this.connection().status !== ConnectionStatus.IDLE;
    }
    protected get reverse(): boolean {
        return this.connection().status === ConnectionStatus.REVERSE;
    }
    protected get node1(): Position {
        return this.connection().node1.position;
    }
    protected get node2(): Position {
        return this.connection().node2.position;
    }
    protected get w(): number {
        return Math.abs(this.node2.x - this.node1.x) + 60;
    }
    protected get h(): number {
        return Math.abs(this.node2.y - this.node1.y) + 60;
    }
    protected get t(): Position {
        return {
            x: (this.node1.x + this.node2.x) / 2 - this.w / 2,
            y: (this.node1.y + this.node2.y) / 2 + this.h / 2,
        };
    }

    protected onClick(): void {
        this.dialog.open(EditConnectionDialogComponent, {
            context: { connection: this.connection() },
        });
    }
}
