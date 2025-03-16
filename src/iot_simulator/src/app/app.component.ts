import { Component } from "@angular/core";
import { Router, RouterOutlet } from "@angular/router";
import { CanvasComponent } from "@components/canvas/canvas.component";
import { ConnectionComponent } from "@components/connection/connection.component";
import { MenuBarComponent } from "@components/menu-bar/menu-bar.component";
import { NodeComponent } from "@components/node/node.component";
import { Connection } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideChevronsLeftRightEllipsis,
    lucideGhost,
    lucideSettings,
    lucideTrash,
} from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnContextMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemDirective,
    HlmMenuItemIconDirective,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
} from "@spartan-ng/ui-menu-helm";
import { HlmToasterComponent } from "@spartan-ng/ui-sonner-helm";
import { floatAnimation } from "./app.routes.transition";

@Component({ template: "" })
export class BlankComponent {}

@Component({
    selector: "app-root",
    imports: [
        RouterOutlet,
        BrnContextMenuTriggerDirective,
        CanvasComponent,
        ConnectionComponent,
        HlmMenuComponent,
        HlmMenuGroupComponent,
        HlmMenuItemDirective,
        HlmMenuItemIconDirective,
        HlmMenuSeparatorComponent,
        HlmMenuShortcutComponent,
        HlmToasterComponent,
        MenuBarComponent,
        NodeComponent,
        NgIcon,
    ],
    providers: [
        provideIcons({
            lucideChevronsLeftRightEllipsis,
            lucideGhost,
            lucideSettings,
            lucideTrash,
        }),
    ],
    templateUrl: "app.component.html",
    animations: [floatAnimation],
})
export class AppComponent {
    protected get nodes(): Node[] {
        return this._networkManager.nodes;
    }
    protected get connections(): Connection[] {
        return this._networkManager.router?.connections ?? [];
    }

    public constructor(
        private readonly _router: Router,
        private readonly _networkManager: NetworkManagerService,
    ) {}

    protected getAnimationData(outlet: RouterOutlet) {
        return (
            (outlet &&
                outlet.isActivated &&
                outlet.activatedRoute.snapshot.params["mac"]) ||
            "_"
        );
    }

    protected onInsertNode(event: Event, type?: string) {
        event.preventDefault();
        if (event instanceof MouseEvent) {
            const { clientX, clientY } = event as MouseEvent;
        }

        this._networkManager.addNode(type as NodeType);
    }

    protected onDeleteNode(event: Event, mac: string) {
        event.preventDefault();

        this._networkManager.deleteNode(mac);
    }

    protected onNodeTraffic(event: Event, mac: string) {
        event.preventDefault();

        this._router.navigate([mac, "network-traffic"]);
    }

    protected onNodeAttack(event: Event, mac: string) {
        event.preventDefault();

        this._router.navigate([mac, "attack"]);
    }

    protected onNodeConfig(event: Event, mac: string) {
        event.preventDefault();

        this._router.navigate([mac, "configuration"]);
    }
}
