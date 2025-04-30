import {
    Component,
    inject,
    output,
    OutputEmitterRef,
    Signal,
    WritableSignal
} from "@angular/core";
import { TranslateModule } from "@ngx-translate/core";
import { ConfigService } from "@services/config.service";
import { NetworkService } from "@services/network.service";
import { StateService } from "@services/state.service";
import { BrnMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuBarComponent,
    HlmMenuBarItemDirective,
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemCheckboxDirective,
    HlmMenuItemCheckComponent,
    HlmMenuItemDirective,
    HlmMenuItemSubIndicatorComponent,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
    HlmSubMenuComponent,
} from "@spartan-ng/ui-menu-helm";

@Component({
    selector: "app-menu-bar",
    imports: [
        TranslateModule,
        BrnMenuTriggerDirective,
        HlmMenuComponent,
        HlmMenuBarComponent,
        HlmSubMenuComponent,
        HlmMenuItemDirective,
        HlmMenuItemSubIndicatorComponent,
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
    public readonly config: ConfigService = inject(ConfigService);
    public readonly state: StateService = inject(StateService);
    public readonly network: NetworkService = inject(NetworkService);
    protected readonly onNewFile: OutputEmitterRef<void> = output();
    protected readonly onOpenFile: OutputEmitterRef<void> = output();
    protected readonly onLoadExternalLibrary: OutputEmitterRef<void> = output();
    protected readonly onLoadModels: OutputEmitterRef<void> = output();
    protected readonly onSaveFile: OutputEmitterRef<void> = output();
    protected readonly onUndo: OutputEmitterRef<void> = output();
    protected get canUndo(): boolean {
        return this.state.canUndo;
    }
    protected readonly onRedo: OutputEmitterRef<void> = output();
    protected get canRedo(): boolean {
        return this.state.canRedo;
    }
    protected readonly onInsertRouter: OutputEmitterRef<void> = output();
    protected readonly onInsertDevice: OutputEmitterRef<void> = output();
    protected get canInsertRouter(): boolean {
        return !this.network.router;
    }
    protected readonly language: WritableSignal<string> = this.config.language;
    protected readonly showGrid: Signal<boolean> = this.config.grid;
    protected readonly onCenter: OutputEmitterRef<void> = output();
    protected readonly onResetZoom: OutputEmitterRef<void> = output();
    protected readonly onZoomIn: OutputEmitterRef<void> = output();
    protected readonly onZoomOut: OutputEmitterRef<void> = output();

    protected handleOnNewFile() {
        this.onNewFile.emit();
    }

    protected handleOnOpenFile() {
        this.onOpenFile.emit();
    }

    protected handleOnLoadExternalLibrary() {
        this.onLoadExternalLibrary.emit();
    }

    protected handleOnLoadModels() {
        this.onLoadModels.emit();
    }

    protected handleOnSaveFile() {
        this.onSaveFile.emit();
    }

    protected handleOnUndo() {
        this.onUndo.emit();
    }

    protected handleOnRedo() {
        this.onRedo.emit();
    }

    protected handleOnInsertRouter() {
        this.onInsertRouter.emit();
    }

    protected handleOnInsertDevice() {
        this.onInsertDevice.emit();
    }

    protected handleOnLanguage(language: string) {
        this.language.set(language);
    }

    protected handleOnGrid() {
        this.config.grid.update((value) => !value);
    }

    protected handleOnCenter() {
        this.onCenter.emit();
    }

    protected handleOnResetZoom() {
        this.onResetZoom.emit();
    }

    protected handleOnZoomIn() {
        this.onZoomIn.emit();
    }

    protected handleOnZoomOut() {
        this.onZoomOut.emit();
    }
}
