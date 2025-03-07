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
        return this._networkManager.connections;
    }

    public constructor(
        private _router: Router,
        private _networkManager: NetworkManagerService,
    ) {}

    protected addNode({ clientX, clientY }: MouseEvent, type: string) {
        this._networkManager.addNode(type as NodeType);
    }

    protected deleteNode(mac: string) {
        this._networkManager.deleteNode(mac);
    }

    protected nodeTraffic(mac: string) {
        this._router.navigate([mac, "network-traffic"]);
    }

    protected nodeConfig(mac: string) {
        this._router.navigate([mac, "configuration"]);
    }
}
