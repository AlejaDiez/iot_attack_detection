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
    protected get canInsertRouter(): boolean {
        return this._networkManager.router === undefined;
    }

    public constructor(
        private readonly _networkManager: NetworkManagerService,
    ) {}

    @HostListener("document:keydown.meta.n", ["$event"])
    protected onNew(event: Event) {
        event.preventDefault();
        this._networkManager.new();
    }

    @HostListener("document:keydown.meta.o", ["$event"])
    protected onOpen(event: Event) {
        event.preventDefault();
        this._networkManager.loadFromFile();
    }

    @HostListener("document:keydown.meta.w", ["$event"])
    protected onClose(event: Event) {
        event.preventDefault();
        window.close();
    }

    @HostListener("document:keydown.meta.s", ["$event"])
    protected onSave(event: Event) {
        event.preventDefault();
        this._networkManager.saveToFile();
    }

    @HostListener("document:keydown.meta.z", ["$event"])
    protected onUndo(event: Event) {
        event.preventDefault();
    }

    @HostListener("document:keydown.meta.shift.z", ["$event"])
    protected onRedo(event: Event) {
        event.preventDefault();
    }

    @HostListener("document:keydown.meta.shift.d", ["$event"])
    @HostListener("document:keydown.meta.shift.r", ["$event"])
    protected onInsertNode(event: Event, type?: string) {
        event.preventDefault();
        if (event instanceof KeyboardEvent)
            type = (event as KeyboardEvent).key === "r" ? "router" : undefined;
        this._networkManager.addNode(type as NodeType);
    }
}
