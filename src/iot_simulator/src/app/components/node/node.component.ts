import { CommonModule } from "@angular/common";
import {
    AfterViewInit,
    Component,
    ElementRef,
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
export class NodeComponent implements AfterViewInit, OnDestroy {
    @ViewChild("id")
    id!: ElementRef<HTMLDivElement>;
    @Input()
    node!: Device;
    moving: boolean = false;

    private mouseDown(event: MouseEvent): void {
        if (event.button === 0) {
            this.moving = true;
        }
    }

    private mouseMove(event: MouseEvent): void {
        if (this.moving) {
            this.node.move({
                x: this.node.position.x + event.movementX,
                y: this.node.position.y - event.movementY,
            });
        }
    }

    private mouseUp(event: MouseEvent): void {
        if (this.moving) {
            this.moving = false;
            this.node.move({
                x: Math.round(this.node.position.x / 20) * 20,
                y: Math.round(this.node.position.y / 20) * 20,
            });
        }
    }

    ngAfterViewInit(): void {
        this.id.nativeElement.addEventListener(
            "mousedown",
            this.mouseDown.bind(this)
        );
        document.addEventListener("mousemove", this.mouseMove.bind(this));
        document.addEventListener("mouseup", this.mouseUp.bind(this));
    }

    ngOnDestroy(): void {
        this.id.nativeElement.removeEventListener(
            "mousedown",
            this.mouseDown.bind(this)
        );
        document.removeEventListener("mousemove", this.mouseMove.bind(this));
        document.removeEventListener("mouseup", this.mouseUp.bind(this));
    }
}
