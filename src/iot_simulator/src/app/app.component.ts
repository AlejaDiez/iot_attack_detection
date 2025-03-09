import { CommonModule } from "@angular/common";
import { Component } from "@angular/core";
import { Router, RouterOutlet } from "@angular/router";
import { CanvasComponent } from "@components/canvas/canvas.component";
import { ConnectionComponent } from "@components/connection/connection.component";
import { MenuBarComponent } from "@components/menu-bar/menu-bar.component";
import { NodeComponent } from "@components/node/node.component";
import { Connection } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnContextMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemDirective,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
} from "@spartan-ng/ui-menu-helm";
import { HlmToasterComponent } from "@spartan-ng/ui-sonner-helm";

@Component({
    selector: "app-root",
    imports: [
        CommonModule,
        RouterOutlet,
        BrnContextMenuTriggerDirective,
        CanvasComponent,
        ConnectionComponent,
        HlmMenuComponent,
        HlmMenuGroupComponent,
        HlmMenuItemDirective,
        HlmMenuSeparatorComponent,
        HlmMenuShortcutComponent,
        HlmToasterComponent,
        MenuBarComponent,
        NodeComponent,
    ],
    templateUrl: "app.component.html",
})
export class AppComponent {
    protected get nodes(): Node[] {
        return this._networkManager.nodes;
    }
    protected get connections(): Connection[] {
        return this._networkManager.connections ?? [];
    }

    public constructor(
        private _router: Router,
        private _networkManager: NetworkManagerService,
    ) {}

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

    protected onNodeConfig(event: Event, mac: string) {
        event.preventDefault();

        this._router.navigate([mac, "configuration"]);
    }
}
