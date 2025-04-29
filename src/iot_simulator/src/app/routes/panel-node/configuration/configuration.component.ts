import { CommonModule } from "@angular/common";
import { Component, inject, input, InputSignal } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { HlmButtonModule } from "@components/ui/ui-button-helm/src";
import { Node, NodeType } from "@models/node";
import { NetworkService } from "@services/network.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmSwitchModule } from "@spartan-ng/ui-switch-helm";

@Component({
    imports: [
        CommonModule,
        FormsModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmInputModule,
        HlmLabelModule,
        HlmSelectModule,
        HlmSwitchModule,
    ],
    templateUrl: "configuration.component.html",
    host: { class: "flex flex-col gap-4" },
})
export class ConfigurationComponent {
    public readonly network: NetworkService = inject(NetworkService);
    public readonly NodeType: typeof NodeType = NodeType;
    protected readonly node: InputSignal<Node> = input.required<Node>();

    protected delete(): void {
        this.network.deleteNode(this.node().mac);
    }
}
