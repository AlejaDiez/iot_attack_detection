import { CommonModule } from "@angular/common";
import { Component, HostListener, Input } from "@angular/core";
import { RouterModule } from "@angular/router";
import { Node } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideCpu, lucideLaptop, lucideRouter } from "@ng-icons/lucide";
import { HlmCardModule } from "@spartan-ng/ui-card-helm";

@Component({
    selector: "app-node",
    imports: [CommonModule, HlmCardModule, NgIcon, RouterModule],
    providers: [provideIcons({ lucideRouter, lucideCpu, lucideLaptop })],
    templateUrl: "node.component.html",
    styleUrl: "node.component.css",
})
export class NodeComponent {
    protected dragging: boolean = false;
    @Input({ required: true })
    public node!: Node;

    @HostListener("mousedown", ["$event"])
    private _mouseDown(event: MouseEvent): void {
        if (event.button === 0) {
            this.dragging = true;
        }
    }

    @HostListener("document:mousemove", ["$event"])
    private _mouseMove(event: MouseEvent): void {
        if (this.dragging) {
            this.node.move(event.movementX, -event.movementY);
        }
    }

    @HostListener("document:mouseup", ["$event"])
    private _mouseUp(event: MouseEvent): void {
        if (this.dragging) {
            this.dragging = false;
            this.node.moveTo(
                Math.round(this.node.position.x / 20) * 20,
                Math.round(this.node.position.y / 20) * 20
            );
        }
    }
}
