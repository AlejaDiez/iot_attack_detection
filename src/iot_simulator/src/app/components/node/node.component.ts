import { CommonModule } from "@angular/common";
import { Component, HostListener, Input } from "@angular/core";
import { RouterModule } from "@angular/router";
import { Device } from "@models/device";
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
    @Input()
    public node!: Device;
    public moving: boolean = false;
    public active: boolean = false;
    public selected: boolean = false;

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
