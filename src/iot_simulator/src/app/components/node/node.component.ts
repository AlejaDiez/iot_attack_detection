import { CommonModule } from "@angular/common";
import {
    AfterViewInit,
    Component,
    ElementRef,
    HostListener,
    Input,
    OnDestroy,
    ViewChild,
} from "@angular/core";
import { Device } from "../../models/device";

@Component({
    selector: "app-node",
    imports: [CommonModule],
    templateUrl: "node.component.html",
    styleUrl: "node.component.css",
})
export class NodeComponent {
    @Input()
    public node!: Device;
    public moving: boolean = false;

    @HostListener("mousedown", ["$event"])
    private _mouseDown(event: MouseEvent): void {
        if (event.button === 0) {
            this.moving = true;
        }
    }

    @HostListener("document:mousemove", ["$event"])
    private _mouseMove(event: MouseEvent): void {
        if (this.moving) {
            this.node.move({
                x: this.node.position.x + event.movementX,
                y: this.node.position.y - event.movementY,
            });
        }
    }

    @HostListener("document:mouseup", ["$event"])
    private _mouseUp(event: MouseEvent): void {
        if (this.moving) {
            this.moving = false;
            this.node.move({
                x: Math.round(this.node.position.x / 20) * 20,
                y: Math.round(this.node.position.y / 20) * 20,
            });
        }
    }
}
