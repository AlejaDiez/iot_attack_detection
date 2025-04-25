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
import { NetworkService } from "@services/network.service";
import { StateService } from "@services/state.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmSwitchModule } from "@spartan-ng/ui-switch-helm";
import { isEqual } from "lodash";
import { debounceTime, distinctUntilChanged } from "rxjs";

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
    public readonly network: NetworkService = inject(NetworkService);
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

            this.form.patchValue({ name, ip, type }, { emitEvent: false });
            if (NodeType.getTypes(type).length === 1)
                this.form.get("type")!.disable({ emitEvent: false });
            else this.form.get("type")!.enable({ emitEvent: false });
        });
    }

    public ngOnInit(): void {
        this.form.valueChanges.subscribe(({ name, type }) => {
            if (name) this.node().name = name;
            if (type) this.node().type = type;
        });
        // Guardar el estado de la red
        this.form.valueChanges
            .pipe(
                debounceTime(StateService.UPDATE_WAIT),
                distinctUntilChanged((prev, curr) => isEqual(prev, curr)),
            )
            .subscribe(() => this.network.saveState());
    }

    protected delete(): void {
        this.network.deleteNode(this.node().mac);
    }
}
