import { CommonModule } from "@angular/common";
import {
    Component,
    inject,
    input,
    InputSignal,
    output,
    OutputEmitterRef,
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
        CommonModule,
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
    public readonly externalLibrary: InputSignal<boolean> = input.required();
    protected readonly onLoadExternalLibrary: OutputEmitterRef<void> = output();
    protected readonly onDeleteExternalLibrary: OutputEmitterRef<void> =
        output();
    public readonly models: InputSignal<boolean> = input.required();
    protected readonly onLoadModels: OutputEmitterRef<void> = output();
    protected readonly onDeleteModels: OutputEmitterRef<void> = output();
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
    public readonly language: InputSignal<string> = input.required();
    protected readonly onChangeLanguage: OutputEmitterRef<string> = output();
    public readonly highContrast: InputSignal<boolean> = input.required();
    protected onHighContrast: OutputEmitterRef<void> = output();
    public readonly showGrid: InputSignal<boolean> = input.required();
    protected readonly onShowGrid: OutputEmitterRef<void> = output();
    protected readonly onCenter: OutputEmitterRef<void> = output();
    protected readonly onZoomReset: OutputEmitterRef<void> = output();
    protected readonly onZoomIn: OutputEmitterRef<void> = output();
    protected get canZoomIn(): boolean {
        return this.config.zoom() < 2;
    }
    protected readonly onZoomOut: OutputEmitterRef<void> = output();
    protected get canZoomOut(): boolean {
        return this.config.zoom() > 0.5;
    }

    protected handleOnNewFile() {
        this.onNewFile.emit();
    }

    protected handleOnOpenFile() {
        this.onOpenFile.emit();
    }

    protected handleOnLoadExternalLibrary() {
        this.onLoadExternalLibrary.emit();
    }

    protected handleOnDeleteExternalLibrary() {
        this.onDeleteExternalLibrary.emit();
    }

    protected handleOnLoadModels() {
        this.onLoadModels.emit();
    }

    protected handleOnDeleteModels() {
        this.onDeleteModels.emit();
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
        this.onChangeLanguage.emit(language);
    }

    protected handleOnHighContrast() {
        this.onHighContrast.emit();
    }

    protected handleOnGrid() {
        this.onShowGrid.emit();
    }

    protected handleOnCenter() {
        this.onCenter.emit();
    }

    protected handleOnZoomReset() {
        this.onZoomReset.emit();
    }

    protected handleOnZoomIn() {
        this.onZoomIn.emit();
    }

    protected handleOnZoomOut() {
        this.onZoomOut.emit();
    }
}
