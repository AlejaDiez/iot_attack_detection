import { Component, HostListener } from "@angular/core";
import { NodeType } from "@models/node";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuBarComponent,
    HlmMenuBarItemDirective,
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemCheckboxDirective,
    HlmMenuItemCheckComponent,
    HlmMenuItemDirective,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
} from "@spartan-ng/ui-menu-helm";

@Component({
    selector: "app-menu-bar",
    imports: [
        BrnMenuTriggerDirective,
        HlmMenuComponent,
        HlmMenuBarComponent,
        HlmMenuItemDirective,
        HlmMenuShortcutComponent,
        HlmMenuSeparatorComponent,
        HlmMenuBarItemDirective,
        HlmMenuItemCheckComponent,
        HlmMenuGroupComponent,
        HlmMenuItemCheckboxDirective,
    ],
    templateUrl: "menu-bar.component.html",
})
export class MenuBarComponent {
    protected get canInsertRouter() {
        return !this._networkManager.router;
    }

    constructor(private _networkManager: NetworkManagerService) {}

    @HostListener("document:keydown.meta.shift.r", ["$event"])
    protected onInsertRouterClick(event?: Event) {
        event?.preventDefault();
        this._networkManager.addNode(NodeType.ROUTER);
    }

    @HostListener("document:keydown.meta.shift.c", ["$event"])
    protected onInsertComputerClick(event?: Event) {
        event?.preventDefault();
        this._networkManager.addNode(NodeType.COMPUTER);
    }

    @HostListener("document:keydown.meta.shift.i", ["$event"])
    protected onInsertIoTClick(event?: Event) {
        event?.preventDefault();
        this._networkManager.addNode(NodeType.IOT);
    }
}
