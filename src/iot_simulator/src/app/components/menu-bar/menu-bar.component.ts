import { Component, EventEmitter, Output } from "@angular/core";
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
    @Output()
    public onNewFile: EventEmitter<void> = new EventEmitter<void>();
    @Output()
    public onOpenFile: EventEmitter<void> = new EventEmitter<void>();
    @Output()
    public onCloseFile: EventEmitter<void> = new EventEmitter<void>();
    @Output()
    public onSaveFile: EventEmitter<void> = new EventEmitter<void>();
    @Output()
    public onInsertRouter: EventEmitter<void> = new EventEmitter<void>();
    @Output()
    public onInsertDevice: EventEmitter<void> = new EventEmitter<void>();
    protected get canInsertRouter(): boolean {
        return !this._networkManager.router;
    }

    public constructor(
        private readonly _networkManager: NetworkManagerService,
    ) {}

    protected handleOnNewFile() {
        this.onNewFile.emit();
    }

    protected handleOnOpenFile() {
        this.onOpenFile.emit();
    }

    protected handleOnCloseFile() {
        this.onCloseFile.emit();
    }

    protected handleOnSaveFile() {
        this.onSaveFile.emit();
    }

    protected handleOnInsertRouter() {
        this.onInsertRouter.emit();
    }

    protected handleOnInsertDevice() {
        this.onInsertDevice.emit();
    }
}
