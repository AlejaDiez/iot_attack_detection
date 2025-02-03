import { CommonModule } from "@angular/common";
import {
    AfterViewInit,
    Component,
    ElementRef,
    Input,
    OnDestroy,
    ViewChild,
} from "@angular/core";

@Component({
    selector: "app-node",
    imports: [CommonModule],
    templateUrl: "node.component.html",
    styleUrl: "node.component.css",
})
export class NodeComponent implements AfterViewInit, OnDestroy {
    @ViewChild("node")
    node!: ElementRef<HTMLDivElement>;
    @Input()
    name: string = "";
    @Input()
    ip: string = "";
    @Input()
    type: string = "router";
    x: number = Math.random() * (300 - -300) + -300;
    y: number = Math.random() * (300 - -300) + -300;
    moving: boolean = false;

    private mouseDown(event: MouseEvent): void {
        if (event.button === 0) {
            this.moving = true;
        }
    }

    private mouseMove(event: MouseEvent): void {
        if (this.moving) {
            this.x += event.movementX;
            this.y -= event.movementY;
        }
    }

    private mouseUp(event: MouseEvent): void {
        if (this.moving) {
            this.moving = false;
            this.x = Math.round(this.x / 20) * 20;
            this.y = Math.round(this.y / 20) * 20;
        }
    }

    ngAfterViewInit(): void {
        this.node.nativeElement.addEventListener(
            "mousedown",
            this.mouseDown.bind(this)
        );
        document.addEventListener("mousemove", this.mouseMove.bind(this));
        document.addEventListener("mouseup", this.mouseUp.bind(this));
    }

    ngOnDestroy(): void {
        this.node.nativeElement.removeEventListener(
            "mousedown",
            this.mouseDown.bind(this)
        );
        document.removeEventListener("mousemove", this.mouseMove.bind(this));
        document.removeEventListener("mouseup", this.mouseUp.bind(this));
    }
}
