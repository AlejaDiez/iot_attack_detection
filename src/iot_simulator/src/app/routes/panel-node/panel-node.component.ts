import { Component, input, InputSignal } from "@angular/core";
import { RouterModule } from "@angular/router";
import { Node, NodeType } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideGhost, lucideX } from "@ng-icons/lucide";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmCardDirective, HlmCardModule } from "@spartan-ng/ui-card-helm";

@Component({
    selector: "app-panel-node",
    imports: [HlmButtonModule, HlmCardModule, NgIcon, RouterModule],
    providers: [provideIcons({ lucideGhost, lucideX })],
    host: {
        class: "fixed right-5 top-1/2 flex flex-col max-h-[calc(100%-2.5rem)] min-w-[512px] max-w-[calc(100%-2.5rem)] -translate-y-1/2 rounded-xl overflow-hidden",
    },
    hostDirectives: [HlmCardDirective],
    templateUrl: "panel-node.component.html",
})
export class PanelNodeComponent {
    protected readonly node: InputSignal<Node> = input.required<Node>();
    protected readonly NodeType: typeof NodeType = NodeType;
}
