import { Component } from "@angular/core";
import { HlmButtonDirective } from "@spartan-ng/ui-button-helm";
import { BrnMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuBarComponent,
    HlmMenuBarItemDirective,
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemCheckboxDirective,
    HlmMenuItemCheckComponent,
    HlmMenuItemDirective,
    HlmMenuItemIconDirective,
    HlmMenuItemRadioComponent,
    HlmMenuItemRadioDirective,
    HlmMenuItemSubIndicatorComponent,
    HlmMenuLabelComponent,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
    HlmSubMenuComponent,
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
    templateUrl: "./menu-bar.component.html",
})
export class MenuBarComponent {}
