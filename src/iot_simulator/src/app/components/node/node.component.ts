import { CommonModule } from "@angular/common";
import {
    Component,
    HostListener,
    inject,
    input,
    InputSignal,
} from "@angular/core";
import { Router, RouterModule } from "@angular/router";
import { Node, NodeType } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideCpu,
    lucideGhost,
    lucideLaptop,
    lucideRouter,
    lucideShield,
} from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { HlmCardModule } from "@spartan-ng/ui-card-helm";

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
    host: { class: "contents" },
})
export class NodeComponent {
    public readonly router: Router = inject(Router);
    public readonly networkManager: NetworkManagerService = inject(
        NetworkManagerService,
    );
    public readonly NodeType: typeof NodeType = NodeType;
    public readonly node: InputSignal<Node> = input.required();
    protected get focused(): boolean {
        return this.networkManager.focusedNode === this.node().mac;
    }
    protected clicked: boolean = false;
    protected dragging: boolean = false;

    @HostListener("mousedown", ["$event"])
    private _mouseDown(event: MouseEvent): void {
        event.preventDefault();
        if (event.button === 0) this.clicked = true;
    }

    @HostListener("document:mousemove", ["$event"])
    private _mouseMove(event: MouseEvent): void {
        event.preventDefault();
        if (
            this.clicked &&
            Math.sqrt(event.movementX ** 2 + event.movementY ** 2) > 1
        ) {
            this.dragging = true;
            this.router.navigate([""]);
        }
        if (this.dragging) {
            this.clicked = false;
            this.node().move(event.movementX, -event.movementY, true);
        }
    }

    @HostListener("document:mouseup", ["$event"])
    private _mouseUp(event: MouseEvent): void {
        event.preventDefault();
        if (this.clicked) {
            if (this.focused) this.router.navigate([""]);
            else {
                const route =
                    this.router.url.split("/").length > 2
                        ? this.router.url.split("/")[2]
                        : "network-traffic";

                this.router.navigate([this.node().mac, route]);
            }
        }
        if (this.dragging)
            this.node().move(
                Math.round(this.node().position.x / 20) * 20,
                Math.round(this.node().position.y / 20) * 20,
            );
        this.clicked = false;
        this.dragging = false;
    }

    @HostListener("document:keydown.delete", ["$event"])
    @HostListener("document:keydown.meta.backspace", ["$event"])
    private _supr(event: KeyboardEvent): void {
        event.preventDefault();
        if (this.focused) this.networkManager.deleteNode(this.node().mac);
    }
}
