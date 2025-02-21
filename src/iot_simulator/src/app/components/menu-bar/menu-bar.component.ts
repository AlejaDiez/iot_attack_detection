import { Component } from "@angular/core";
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
    templateUrl: "./menu-bar.component.html",
})
export class MenuBarComponent {}
