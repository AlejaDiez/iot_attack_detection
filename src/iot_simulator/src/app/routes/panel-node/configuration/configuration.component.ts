import { CommonModule } from "@angular/common";
import { Component, inject, input, InputSignal } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { HlmButtonImports } from "@components/ui/ui-button-helm/src";
import { HlmInputImports } from "@components/ui/ui-input-helm/src";
import { HlmLabelImports } from "@components/ui/ui-label-helm/src";
import { HlmSelectImports } from "@components/ui/ui-select-helm/src";
import { HlmSwitchImports } from "@components/ui/ui-switch-helm/src";
import { Node, NodeType } from "@models/node";
import { TranslateModule } from "@ngx-translate/core";
import { ConfigService } from "@services/config.service";
import { NetworkService } from "@services/network.service";
import { BrnSelectImports } from "@spartan-ng/brain/select";

@Component({
    imports: [
        CommonModule,
        FormsModule,
        TranslateModule,
        BrnSelectImports,
        HlmButtonImports,
        HlmInputImports,
        HlmLabelImports,
        HlmSelectImports,
        HlmSwitchImports,
    ],
    templateUrl: "configuration.component.html",
    host: { class: "flex flex-col gap-4 pb-6" },
})
export class ConfigurationComponent {
    public readonly config: ConfigService = inject(ConfigService);
    public readonly network: NetworkService = inject(NetworkService);
    public readonly NodeType: typeof NodeType = NodeType;
    protected readonly node: InputSignal<Node> = input.required<Node>();

    protected delete(): void {
        this.network.deleteNode(this.node().mac);
    }
}
