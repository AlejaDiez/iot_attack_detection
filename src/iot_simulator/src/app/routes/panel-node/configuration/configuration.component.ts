import { CommonModule } from "@angular/common";
import {
    Component,
    effect,
    inject,
    input,
    InputSignal,
    OnInit,
} from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { HlmButtonModule } from "@components/ui/ui-button-helm/src";
import { Node, NodeType } from "@models/node";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmSwitchModule } from "@spartan-ng/ui-switch-helm";

@Component({
    imports: [
        CommonModule,
        ReactiveFormsModule,
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
export class ConfigurationComponent implements OnInit {
    public readonly networkManager: NetworkManagerService = inject(
        NetworkManagerService,
    );
    public readonly NodeType: typeof NodeType = NodeType;
    protected readonly node: InputSignal<Node> = input.required<Node>();
    protected form: FormGroup = new FormGroup({
        name: new FormControl(null),
        ip: new FormControl({ value: null, disabled: true }),
        type: new FormControl(null),
    });

    public constructor() {
        effect(() => {
            const { name, ip, type } = this.node();

            this.form.patchValue({ name, ip, type });
            if (NodeType.getTypes(type).length === 1)
                this.form.get("type")!.disable();
            else this.form.get("type")!.enable();
        });
    }

    public ngOnInit(): void {
        this.form.valueChanges.subscribe(({ name, type }) => {
            if (name) this.node().name = name;
            if (type) this.node().type = type;
        });
    }

    protected delete(): void {
        this.networkManager.deleteNode(this.node().mac);
    }
}
