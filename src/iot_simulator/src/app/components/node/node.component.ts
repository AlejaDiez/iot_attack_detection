import { CommonModule } from "@angular/common";
import { Component, HostListener, Input } from "@angular/core";
import { NavigationEnd, Router, RouterModule } from "@angular/router";
import { Node, NodeType } from "@models/node";
import { Position } from "@models/position";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideCpu,
    lucideGhost,
    lucideLaptop,
    lucideRouter,
    lucideShield,
} from "@ng-icons/lucide";
import { HlmCardModule } from "@spartan-ng/ui-card-helm";
import { filter } from "rxjs";

@Component({
    selector: "app-node",
    imports: [CommonModule, HlmCardModule, NgIcon, RouterModule],
    providers: [
        provideIcons({
            lucideCpu,
            lucideGhost,
            lucideLaptop,
            lucideRouter,
            lucideShield,
        }),
    ],
    templateUrl: "node.component.html",
    styleUrl: "node.component.css",
})
export class NodeComponent {
    @Input({ required: true })
    public node!: Node;
    protected get ip(): string | undefined {
        return this.node.ip;
    }
    protected get name(): string {
        return this.node.name;
    }
    protected get type(): NodeType {
        return this.node.type;
    }
    protected focused: boolean = false;
    protected clicked: boolean = false;
    protected dragging: boolean = false;
    protected get position(): Position {
        return this.node.position;
    }
    protected get communicating(): boolean {
        return this.node.communicating;
    }
    protected get cyberShieldActive(): boolean {
        return this.node.type === NodeType.ROUTER;
    }
    protected get phantomAttackerActive(): boolean {
        return this.node.type === NodeType.COMPUTER;
    }

    public constructor(private readonly _router: Router) {
        this._router.events
            .pipe(filter((event) => event instanceof NavigationEnd))
            .subscribe(
                ({ url }) => (this.focused = url.includes(this.node.mac)),
            );
    }

    @HostListener("mousedown", ["$event"])
    private _mouseDown(event: MouseEvent): void {
        if (event.button === 0) this.clicked = true;
    }

    @HostListener("document:mousemove", ["$event"])
    private _mouseMove(event: MouseEvent): void {
        if (
            this.clicked &&
            Math.sqrt(event.movementX ** 2 + event.movementY ** 2) > 1
        ) {
            this.dragging = true;
            this._router.navigate([""]);
        }
        if (this.dragging) {
            this.clicked = false;
            this.node.move(event.movementX, -event.movementY, true);
        }
    }

    @HostListener("document:mouseup", ["$event"])
    private _mouseUp(event: MouseEvent): void {
        if (this.clicked) {
            if (this.focused) this._router.navigate([""]);
            else this._router.navigate([this.node.mac, "network-traffic"]);
        }
        if (this.dragging) {
            this.node.move(
                Math.round(this.node.position.x / 20) * 20,
                Math.round(this.node.position.y / 20) * 20,
            );
        }
        this.clicked = false;
        this.dragging = false;
    }
}
