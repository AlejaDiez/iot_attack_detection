import { CommonModule } from "@angular/common";
import { Component, inject, input, InputSignal, OnInit } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { Router } from "@angular/router";
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
    public readonly router: Router = inject(Router);
    public readonly networkManager: NetworkManagerService = inject(
        NetworkManagerService,
    );
    public readonly NodeType = NodeType;
    protected readonly node: InputSignal<Node> = input.required<Node>();
    protected form: FormGroup = new FormGroup({
        name: new FormControl(null),
        ip: new FormControl({ value: null, disabled: true }),
        type: new FormControl(null),
    });

    public ngOnInit(): void {
        this.form.patchValue({
            name: this.node().name,
            ip: this.node().ip,
            type: this.node().type,
        });
        this.form.valueChanges.subscribe((value) => {
            this.node().name = value.name;
            this.node().type = value.type;
        });
    }

    protected delete(): void {
        this.networkManager
            .deleteNode(this.node().mac)
            .then((deleted: boolean) =>
                deleted ? this.router.navigate([""]) : null,
            );
    }
}
