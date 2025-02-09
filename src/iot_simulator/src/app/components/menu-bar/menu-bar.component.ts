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
        HlmSubMenuComponent,
        HlmMenuItemDirective,
        HlmMenuItemSubIndicatorComponent,
        HlmMenuLabelComponent,
        HlmMenuShortcutComponent,
        HlmMenuSeparatorComponent,
        HlmMenuItemIconDirective,
        HlmMenuBarItemDirective,
        HlmMenuItemCheckComponent,
        HlmMenuItemRadioComponent,
        HlmMenuGroupComponent,

        HlmButtonDirective,
        HlmMenuItemCheckboxDirective,
        HlmMenuItemRadioDirective,
    ],
    templateUrl: "./menu-bar.component.html",
})
export class MenuBarComponent {}
